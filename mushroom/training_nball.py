import decimal
import numpy as np
import copy
import operator
from collections import defaultdict
from mushroom.config import Precision, defaultWidthOfPLC, defaultAddDim, defaultSubspaceDim
from mushroom.util import vec_norm, vec_cos, vec_point, qsr_P_degree, qsr_DC, qsr_DC_degree, rotate, get_subspace_code

decimal.getcontext().prec = Precision

DIM_SUBSPACE = defaultSubspaceDim #0
DIM_ADDED = defaultAddDim #50
DIM_PLC = defaultWidthOfPLC #30


def training_one_family_multi_relations(treeStruc=defaultdict(list), root='', w2vDic=dict(), catMemDic=dict(),
                                        subspaceDim = 20,
                                        catFPDic=dict(), addDim=[], ballDict=dict(), L0=0, R0=0, cgap=1, logFile=""):
    spaceChildrenLst = treeStruc[root]
    if len(spaceChildrenLst) > 0:
        for subspace, children in spaceChildrenLst:
            if len(children)> 0:
                for child in children:
                    ballDict = training_one_family_multi_relations(treeStruc=treeStruc, root=child, w2vDic=w2vDic,
                                                               catFPDic=catFPDic, catMemDic=catMemDic, addDim=addDim,
                                                               ballDict=ballDict, L0=L0, R0=R0, logFile=logFile)

                if len(children) > 1:
                    ballDict = training_DC_by_name(children, ballDict=ballDict, catMemDic=catMemDic, logFile=logFile)

                ballDict = making_ball_contained_in(root, children, ballDict=ballDict, w2vDic=w2vDic, catFPDic=catFPDic,
                                            subspace = subspace,  subspaceDim = subspaceDim,
                                            catMemDic=catMemDic, addDim=addDim, L0=L0, R0=R0, cgap=cgap,
                                            logFile=logFile)
                return ballDict
        if len(spaceChildrenLst) > 1:
            # separate sub-spaces
            ballDict = training_DC_subspaces(root, spaceChildrenLst, ballDict=ballDict, catMemDic=catMemDic, logFile=logFile)
        # update root ball to contain all subspaces
        return ballDict

    else:
        ballDict = initialize_ball(root, w2vDic=w2vDic, ballDict=ballDict, catFPDic=catFPDic,
                                   subspaceDim = subspaceDim, subspace='',
                                   addDim=addDim, L0=L0, R0=R0)
        return ballDict


def training_one_family(treeStruc=defaultdict(list), root=None, w2vDic=dict(), catMemDic=dict(), catFPDic=dict(),
                        addDim=[], ballDict=dict(), L0=0, R0=0, cgap=1, logFile=""):
    children = treeStruc[root]
    if root in children:
        print('break')
    if len(children) > 0:
        for child in children:
            ballDict = training_one_family(treeStruc=treeStruc, root=child, w2vDic=w2vDic, catFPDic=catFPDic,
                                           catMemDic=catMemDic, addDim=addDim, ballDict=ballDict, L0=L0, R0=R0,
                                           logFile=logFile)
        # children shall be separated
        if len(children) > 1:
            # print('training dc of root', root)
            ballDict = training_DC_by_name(children, ballDict=ballDict, catMemDic=catMemDic, logFile=logFile)
        # root ball shall contain all children balls
        ballDict = making_ball_contains(root, children, ballDict=ballDict, w2vDic=w2vDic, catFPDic=catFPDic,
                                        catMemDic=catMemDic, addDim=addDim, L0=L0, R0=R0, cgap=cgap, logFile=logFile)
        return ballDict
    else:
        ballDict = initialize_ball(root, w2vDic=w2vDic, ballDict=ballDict, catFPDic=catFPDic,
                                   addDim=addDim, L0=L0, R0=R0)
        return ballDict


def making_ball_contains(root, children, ballDict=None, w2vDic= None, catFPDic=None, catMemDic=None,
                         subspaceDim=0, addDim=[], L0=0, R0=0, cgap=0.1, logFile=""):
    """
    :param root:
    :param children:
    :param ballDict:
    :param addDim:
    :param logFile:
    :return:
    """
    maxL = -1
    flag = False
    while not flag:
        flag = True
        for childName in children:
            ballDict = training_P_by_name(childName, root, ballDict=ballDict,  w2vDic= w2vDic, catFPDic=catFPDic,
                                          catMemDic=catMemDic, subspaceDim=subspaceDim, addDim=addDim, L0=L0, R0=R0,
                                          cgap=cgap, logFile=logFile)
            pBall = ballDict[root]
            if maxL == -1:  # initialize maxL, minL_R
                maxL, minL_R = pBall[-2], decimal.Decimal(pBall[-2]) - decimal.Decimal(pBall[-1])
            if maxL < pBall[-2]:
                maxL = pBall[-2]

            delta = decimal.Decimal(pBall[-2]) - decimal.Decimal(pBall[-1])
            if delta <= 0:
                print('Shifting...', root)
                with open(logFile, 'a+') as wlog:
                    wlog.write(" ".join(["shifting", str(root)] +
                                        [str(ele) for ele in ballDict[root][:-2]] + [str(-delta)]))
                    wlog.write("\n")
                ballDict=shift_whole_tree_of(root, ballDict[root][:-2], -delta, ballDict=ballDict, logFile=logFile)
                flag = False
                break
            elif decimal.Decimal(pBall[-2]) - decimal.Decimal(pBall[-1]) < minL_R:
                minL_R = decimal.Decimal(pBall[-2]) - decimal.Decimal(pBall[-1])

            ballDict.update({root: ballDict[root][:-2] + [maxL, maxL - minL_R + decimal.Decimal(cgap)]})

    return ballDict


def making_ball_contained_in(root, children, ballDict=None, w2vDic=None, catFPDic=None,
                             subspace="",  subspaceDim = 20,
                             catMemDic= None, addDim=[], L0= 0, R0= 0, cgap=0.1,
                             logFile=""):
    # maxL = -1

    ballSample = children[0]

    spaceName = '-'.join([root, subspace])
    ballDict = create_virtual_subspace_ball(children, ballDict, root=root, space=subspace,
                                            w2vDic=w2vDic, catFPDic=catFPDic,
                                            subspaceDim=DIM_SUBSPACE, addDim=addDim,
                                            L0=L0, R0=R0,
                                            catMemDic=catMemDic, logFile=logFile)

    # w2vDic[spaceName] = ballDict[spaceName][:len(w2vDic[root])]
    # catFPDic[spaceName] = catFPDic[ballSample]

    """
    flag = False
    while not flag:
        flag = True
        for childName in children:
            ballDict = training_P_by_name(childName, spaceName, ballDict=ballDict, w2vDic=w2vDic, catFPDic=catFPDic,
                                          catMemDic=catMemDic, subspaceDim = subspaceDim,
                                          subspace=subspace, addDim=addDim, L0=L0, R0=R0, cgap=cgap,
                                          logFile=logFile)
            pBall = ballDict[spaceName]
            if maxL == -1:  # initialize maxL, minL_R
                maxL, minL_R = pBall[-2], decimal.Decimal(pBall[-2]) - decimal.Decimal(pBall[-1])
            if maxL < pBall[-2]:
                maxL = pBall[-2]

            delta = decimal.Decimal(pBall[-2]) - decimal.Decimal(pBall[-1])
            if delta <= 0:
                print('Shifting...', spaceName)
                with open(logFile, 'a+') as wlog:
                    wlog.write(" ".join(["shifting", str(spaceName)] +
                                        [str(ele) for ele in ballDict[spaceName][:-2]] + [str(-delta)]))
                    wlog.write("\n")
                ballDict = shift_whole_tree_of(spaceName, ballDict[spaceName][:-2], -delta,
                                               ballDict=ballDict, logFile=logFile)
                flag = False
                break
            elif decimal.Decimal(pBall[-2]) - decimal.Decimal(pBall[-1]) < minL_R:
                minL_R = decimal.Decimal(pBall[-2]) - decimal.Decimal(pBall[-1])

            ballDict.update({spaceName: ballDict[spaceName][:-2] + [maxL, maxL - minL_R + cgap]})
    """

    ballDict = training_P_by_name(spaceName, root, ballDict=ballDict, w2vDic=w2vDic, catFPDic=catFPDic,
                                  catMemDic=catMemDic, subspaceDim=subspaceDim,
                                  subspace="tr_contain1", addDim=addDim, L0=L0, R0=R0, cgap=cgap,
                                  logFile=logFile)

    return ballDict


def initialize_ball(root, ballDict=None,  w2vDic= None, catFPDic=None, subspaceDim = 0, subspace="",
                    addDim=[], L0=0, R0=0):
    """
    :param root:
    :param ballDict:
    :param catFPDic:
    :param addDim:
    :param L0:
    :param R0:
    :return:
    """
    global DIM_ADDED, DIM_PLC, DIM_SUBSPACE
    fingerPrintUpCat = catFPDic[root]

    DIM_SUBSPACE = subspaceDim
    DIM_ADDED = len(addDim)
    DIM_PLC = len(fingerPrintUpCat)

    if root in w2vDic:
        wvec = w2vDic[root]
    else:
        word = root.split('.')[0]
        wvec = w2vDic[word]
    w2v = [decimal.Decimal(ele * 100) for ele in wvec]
    subspaceCode = get_subspace_code(subspace, spacespaceDim=subspaceDim)
    cpoint = w2v + [decimal.Decimal(ele) + 10 for ele in fingerPrintUpCat] + subspaceCode + addDim
    ballDict.update({root: vec_norm(cpoint) + [L0, R0]})
    return ballDict


def training_P_by_name(childName, atreeName, ballDict=None, w2vDic=None, catMemDic=None, catFPDic=None,
                       subspaceDim=0, subspace="", addDim=[], L0=0, R0=0, cgap=1, sep='.', logFile=""):
    """
    :param childName:
    :param root:
    :param ballDict:
    :param w2vDic:
    :param catFPDic:
    :param addDim:
    :param L0:
    :param R0:
    :param cgap:
    :param sep:
    :param logFile:
    :return:
    """
    if childName.split(sep)[0] == atreeName.split(sep)[0] and ('-' not in childName or '-' not in atreeName): #to do
        BallLeaf = ballDict[childName]
        ballDict = initialize_ball(atreeName, ballDict=ballDict, w2vDic= w2vDic, catFPDic=catFPDic,
                                   subspaceDim=subspaceDim, subspace=subspace, addDim=addDim, L0=L0, R0=R0)
        BallParent = ballDict[atreeName]
        LeafO, ParentO = BallLeaf[:-2], BallParent[:-2]
        LeafL, LeafR = BallLeaf[-2], BallLeaf[-1]
        ParentL, ParentR = LeafL + LeafR + decimal.Decimal(cgap), LeafR + LeafR + decimal.Decimal(cgap + cgap)
        BallParent = ParentO + [ParentL, ParentR]
        ballDict.update({atreeName: BallParent})
    else:
        targetsin0 = 0.6
        while True:
            BallLeaf = ballDict[childName]
            ballDict = initialize_ball(atreeName, ballDict=ballDict, w2vDic= w2vDic, catFPDic=catFPDic,
                                       subspaceDim=subspaceDim, subspace=subspace, addDim=addDim,
                                       L0=L0, R0=R0)
            BallParent = ballDict[atreeName]
            LeafO, ParentO = [decimal.Decimal(ele) for ele in BallLeaf[:-2]], \
                             [decimal.Decimal(ele) for ele in BallParent[:-2]]
            LeafL, LeafR = BallLeaf[-2], BallLeaf[-1]
            sin_beta = BallLeaf[-1] / BallLeaf[-2]

            delta = 1 - sin_beta * sin_beta
            if delta < 0:
                delta = 0
            cos_beta = np.sqrt(delta)
            cos_alpha = np.dot(LeafO, ParentO) / np.linalg.norm(LeafO) / np.linalg.norm(ParentO)

            delta = 1 - cos_alpha * cos_alpha
            if delta < 0:
                delta = 0
            sin_alpha = np.sqrt(delta)

            # begin alpha --> xalpha
            xalpha = sin_alpha / 25
            yalpha = np.sqrt(1 - xalpha * xalpha)
            sin_xalpha = decimal.Decimal(xalpha) *  decimal.Decimal(cos_alpha) +  \
                         decimal.Decimal(yalpha) *  decimal.Decimal(sin_alpha)
            delta = 1 - sin_xalpha * sin_xalpha
            if delta < 0: delta = 0
            cos_xalpha = np.sqrt(delta)

            sin_alpha = sin_xalpha
            cos_alpha = cos_xalpha
            # end

            dOO = LeafL * decimal.Decimal(cos_beta)

            cos_alpha_beta = (decimal.Decimal(cos_beta) * decimal.Decimal(cos_alpha)
                              - decimal.Decimal(sin_beta) * decimal.Decimal(sin_alpha))
            if cos_alpha_beta <= 0:
                # shift_one_family(root=childName, targetsin = targetsin0,  outputPath=outputPath)
                L, R = ballDict[childName][-2:]
                print('Shifting...', childName)
                LNew = R / decimal.Decimal(targetsin0)
                with open(logFile, 'a+') as wlog:
                    wlog.write(" ".join(["shifting", str(childName)] +
                                        [str(ele) for ele in ballDict[childName][:-2]] + [str(LNew - L)]))
                    wlog.write("\n")
                ballDict = shift_whole_tree_of(childName, ballDict[childName][:-2], LNew - L, ballDict=ballDict,
                                               catMemDic=catMemDic, cgap=cgap, logFile=logFile)
                targetsin0 *= 0.9
            else:
                break

        ParentL = dOO / cos_alpha_beta
        if ParentL <= 0 or ParentL == np.inf:
            print("assertion -1  training_P, ParentL", childName, atreeName, ParentL)
            return -1

        ParentR = ParentL * (decimal.Decimal(sin_alpha) * decimal.Decimal(cos_beta)
                             + decimal.Decimal(cos_alpha) * decimal.Decimal(sin_beta)) + decimal.Decimal(0.1)
        BallParent = ParentO + [ParentL, ParentR]
        ballDict.update({atreeName: BallParent})

    count = 0
    while qsr_P_degree(ballDict[childName], ballDict[atreeName]) < 0:
        oldParentR, delta = ParentR, 10
        ParentR += decimal.Decimal(2) - qsr_P_degree(ballDict[childName], ballDict[atreeName])
        while oldParentR == ParentR:
            ParentR += delta
            delta *= 10
        BallParent = ParentO + [ParentL, ParentR]
        ballDict.update({atreeName: BallParent})
        count += 1

    return ballDict


def create_virtual_subspace_ball(ballLst, ballDict, root='', space='', w2vDic=None, catFPDic=None, subspaceDim=0,
                                 addDim=[], L0=0, R0=0, catMemDic=None, logFile=''):
    centralPoints = []
    extendedDim = DIM_SUBSPACE + DIM_PLC + DIM_ADDED +2
    ballSample = ballLst[0]
    for bname in ballLst:
        centralPoints.append(ballDict[bname][:-extendedDim])
    meanPoint = vec_norm(np.sum(np.array(centralPoints), axis = 0))
    vCentralPoint = list(meanPoint) + ballDict[ballSample][-extendedDim:-2]
    cpoint = vec_norm(vCentralPoint)
    rootSubspace = '-'.join([root, space])
    rootW2V = vec_norm(w2vDic[root])
    w2vDic[rootSubspace] = w2vDic[root] #meanPoint #vec_norm(np.sum(np.array([rootW2V, meanPoint]), axis = 0)) #
    catFPDic[rootSubspace] = catFPDic[root]
    catMemDic[rootSubspace] = ballLst
    ballDict = initialize_ball(rootSubspace, ballDict=ballDict, w2vDic=w2vDic, catFPDic=catFPDic,
                                subspaceDim=subspaceDim, subspace=space, addDim=addDim, L0=L0, R0=R0)
    ballDict = making_ball_contains(rootSubspace, ballLst, ballDict=ballDict, w2vDic=w2vDic, catFPDic=catFPDic,
                                    catMemDic=catMemDic, addDim=addDim, subspaceDim=DIM_SUBSPACE, L0=L0, R0=L0,
                                    cgap=0.1, logFile=logFile)

    return ballDict


def training_DC_subspaces(root, spaceChildrenLst, ballDict=None, catMemDic=None,  catFPDic=None,  w2vDic=None,
                          addDim=[], L0=0, R0=0,logFile=""):
    spaceMemberDict = defaultdict(list)
    member2SpaceDic = defaultdict()
    spaceBallDict = dict()
    for space, lst in spaceChildrenLst:
        for ele in lst:
            member2SpaceDic[ele].append(space)
    catMemDic[root] = list(spaceMemberDict.keys())

    spaceBallDict.update(ballDict)
    spaceBallDict = training_DC_by_name(list(spaceMemberDict.keys()), ballDict=spaceBallDict,
                                        catMemDic=catMemDic, cgap =1, logFile= logFile)
    ballDict.update(spaceBallDict)
    return ballDict


def training_DC_by_name(childrenNames, ballDict=None, catMemDic=None, cgap=1, logFile=""):
    dic = dict()
    for tree in childrenNames:
        dic[tree] = ballDict[tree][-2]
    dic0 = copy.deepcopy(dic)

    lst = [(item[0], ballDict[item[0]]) for item in sorted(dic.items(), key=operator.itemgetter(1))]

    i = 0
    while i < len(lst) - 1:
        j = i + 1
        refTreeName = lst[i][0]
        while j < len(lst):
            curTreeName = lst[j][0]
            targetsin0 = 0.6
            while not qsr_DC(ballDict[curTreeName], ballDict[refTreeName]):
                ball1 = ballDict[curTreeName]
                l1, r1 = decimal.Decimal(ball1[-2]), decimal.Decimal(ball1[-1])
                k = r1 / l1
                if k == 1:
                    L, R = ballDict[curTreeName][-2:]
                    print('Shifting...', curTreeName)
                    LNew = R / decimal.Decimal(targetsin0)
                    with open(logFile, 'a+') as wlog:
                        wlog.write(" ".join(["shifting", str(tree)] +
                                            [str(ele) for ele in ballDict[tree][:-2]] + [str(LNew - L)]))
                        wlog.write("\n")
                    ballDict=shift_whole_tree_of(tree, ballDict[curTreeName][:-2], LNew - L, ballDict=ballDict,
                                                 catMemDic=catMemDic, cgap=cgap, logFile=logFile)
                    targetsin0 *= 0.9

                ratio0 = ratio_homothetic_DC_transform_by_name(curTreeName, refTreeName, ballDict=ballDict,
                                                               catMemDic=catMemDic, cgap=cgap, logFile=logFile)
                if ratio0 == -1:
                    return -1
            j += 1
        for tree in childrenNames:
            dic[tree] = ballDict[tree][-2]
        lst = [(item[0], ballDict[item[0]]) for item in sorted(dic.items(), key=operator.itemgetter(1))]
        i += 1

    #####
    # homothetic transformation
    #####
    for child in childrenNames:
        ratio = ballDict[child][-2] / decimal.Decimal(dic0[child])
        ballDict = homothetic_recursive_transform_of_decendents_by_name(child, root=child, rate=ratio, ballDict=ballDict,
                                                                        catMemDic=catMemDic, logFile=logFile)

    return ballDict


def shift_whole_tree_of(tree, deltaVec, deltaL, ballDict=None, catMemDic=None, cgap=1, logFile=""):
    children = catMemDic[tree]

    for child in [ele for ele in children if ele in ballDict]:
        shift_whole_tree_of(child, deltaVec, deltaL, ballDict=ballDict, catMemDic=catMemDic,
                            logFile=logFile)

    if tree not in ballDict:
        return ballDict

    l1, r1 = ballDict[tree][-2:]
    l = np.sqrt(l1 * l1 + deltaL * deltaL
                + 2 * l1 * deltaL * vec_cos(deltaVec, ballDict[tree][:-2]))
    newVec = vec_norm(vec_point(ballDict[tree][:-2], l1) + vec_point(deltaVec, deltaL))
    ballDict.update({tree: list(newVec) + [l, r1]})

    i, j, lst = 0, 0, catMemDic[tree]
    for i in range(len(lst) - 1):
        j = i + 1
        while j < len(lst):
            dcDelta = qsr_DC_degree(ballDict[lst[i]], ballDict[lst[j]])
            if dcDelta < 0:
                print(lst[j], lst[i])
                rotate_vector_till_DC(lst[j],lst[i], ballDict= ballDict, logFile=logFile)
            j += 1

    for child in catMemDic[tree]:
        gap = 1
        while True:
            delta = qsr_P_degree(ballDict[child], ballDict[tree])
            if delta < 0:
                gap *= 2
                ballDict[tree][-1] += - delta + gap
            else:
                break

    return ballDict


def ratio_homothetic_DC_transform_by_name(curTree, refTree, ballDict=None, catMemDic=None, cgap=1, logFile=""):
    ball1 =ballDict[curTree]
    l1, r1 = decimal.Decimal(ball1[-2]), decimal.Decimal(ball1[-1])
    ball0 = ballDict[refTree]
    l0, r0 = decimal.Decimal(ball0[-2]), decimal.Decimal(ball0[-1])
    k = r1 / l1
    targetsin0 = 0.6
    while k >= 1:
        print("assertion -1 k=", k)
        L, R = ballDict[curTree][-2:]

        print('Shifting...', curTree)
        LNew = R / decimal.Decimal(targetsin0)
        with open(logFile, 'a+') as wlog:
            wlog.write(" ".join(["shifting", str(curTree)] +
                                [str(ele) for ele in ballDict[curTree][:-2]] + [str(LNew - L)]))
            wlog.write("\n")
        ballDict = shift_whole_tree_of(curTree, ballDict[curTree][:-2], LNew - L, catMemDic=catMemDic,
                                       gap=cgap, logFile=logFile)
        print('Ended of shifting...', curTree)

        ball1 = ballDict[curTree]
        l1, r1 = decimal.Decimal(ball1[-2]), decimal.Decimal(ball1[-1])
        k = r1 / l1
        targetsin0 *= 0.9

    margin = 10
    while True:
        if ballDict[curTree][-2] == np.inf or ballDict[curTree][-2] < 0:
            print("assertion -1", curTree, "l")
            return -1

        ratio = decimal.Decimal(margin + l0 + r0) / decimal.Decimal(ballDict[curTree][-2] - ballDict[curTree][-1])
        l = ballDict[curTree][-2]
        ballDict[curTree][-2] = l * ratio
        ballDict[curTree][-1] = l * ratio - (l - ballDict[curTree][-1]) * ratio
        delta = qsr_DC_degree(ballDict[curTree], ballDict[refTree])
        if delta > 0:
            break
        decimal.getcontext().prec += 10
        margin *= 10

    print('logging homothetic transformation')
    with open(logFile, 'a+') as wlog:
        wlog.write(" ".join(["homo", str(curTree)] + [str(ratio)]))
        wlog.write("\n")

    return ratio


def homothetic_recursive_transform_of_decendents_by_name(tree, root=None, rate=None, ballDict=None, catMemDic=None,
                                                         logFile=""):
    if rate == 1: return ballDict
    for child in catMemDic[tree]:
        if child not in ballDict: continue
        ballDict=homothetic_recursive_transform_of_decendents_by_name(child, root=root, rate=rate, ballDict=ballDict,
                                                                 catMemDic=catMemDic, logFile=logFile)

        l = decimal.Decimal(ballDict[child][-2]) #tree
        ballDict[child][-2] = l * rate

        if ballDict[child][-2] == np.inf or ballDict[child][-2] < 0:
            print("assertion -1", child, "l")
            return ballDict

        ballDict[child][-1] = l * rate - (l - ballDict[child][-1]) * rate

    # i, j = 0, 0
    lst = catMemDic[tree]
    if len(lst) > 1:
        for i in range(len(lst) - 1):
            j = i + 1
            while j < len(lst):
                dcDelta = qsr_DC_degree(ballDict[lst[i]], ballDict[lst[j]])
                if dcDelta < 0:
                    ballDict = rotate_vector_till_DC(lst[j], lst[i], ballDict=ballDict, logFile=logFile)
                j += 1

    return ballDict


def rotate_vector_till_DC(vecName, vecRefName, ballDict=None, logFile=""):
    dcDelta = qsr_DC_degree(ballDict[vecRefName], ballDict[vecName])
    if dcDelta < 0:
        rotateFlag = True
    rot1 = " ".join(["rotate ", vecName] +
                    [str(ele) for ele in ballDict[vecName][:-1]])
    k = 0
    while dcDelta < 0:
        l1, l2 = ballDict[vecName][-2], ballDict[vecName][-2]
        alpha = (decimal.Decimal(l1*l1)+decimal.Decimal(l2*l2)-decimal.Decimal(dcDelta*dcDelta))/decimal.Decimal(l1*l2*2)
        print('rotating.. ', k, '+/-', alpha)
        v0 = rotate(ballDict[vecName][:-2], alpha)
        v1 = rotate(ballDict[vecName][:-2], -alpha)
        dcDelta0 = qsr_DC_degree(ballDict[vecRefName], v0 + ballDict[vecName][-2:])
        dcDelta1 = qsr_DC_degree(ballDict[vecRefName], v1 + ballDict[vecName][-2:])
        if dcDelta0 >=0:
            ballDict[vecName][:-2] = v0
            dcDelta = dcDelta0
        elif dcDelta1>=0:
            ballDict[vecName][:-2] = v1
            dcDelta = dcDelta1
        k += 1

    if rotateFlag and logFile is not None:
        print('logging rotattion')
        with open(logFile, 'a+') as wlog:
            wlog.write(rot1)
            wlog.write(" ".join(["TO"] + [str(ele) for ele in ballDict[vecName][:-1]]) + "\n")

    return ballDict
