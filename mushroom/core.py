import decimal
import os
import threading
import math
import numpy as np
from collections import defaultdict
from nltk.corpus import wordnet as wn
from copy import deepcopy
from mushroom.config import Precision
from mushroom.training_nball import training_one_family_multi_relations
from mushroom.util import qsr_P, load_train_dataset
from mushroom.util import load_file_to_list, load_entity_vector, create_testing_ball, update_ball_margins

decimal.getcontext().prec = Precision
DEBUG = False

recordEnhencedStem = False
enhencedStems = []

fn_synsets = getattr(wn, 'synsets')
MaxStemLength = 10


def get_logfile_name(h, srel, crel):
    txt = ' '.join([h, srel, crel])
    return ''.join([str(ord(i)) if ord(i) < 97 else i for i in txt])


def create_ball_files(balldic, mushroomFile, outputPath=None):
    lines = []
    for ballname, values in balldic.items():
        lines.append(' '.join([ballname] + [str(ele) for ele in balldic[ballname]]))

    with open(os.path.join(outputPath, mushroomFile), 'w') as bfh:
        bfh.write('\n'.join(lines) + "\n")


def has_e2v(word, e2vec):
    if word in e2vec or word.split('.')[0] in e2vec:
        return True
    else:
        return False


def make_cat_members_from_multi_space_dic(trainingTree):
    catMembers = defaultdict(list)
    for root, space_mlst in trainingTree.items():
        for space, lst in space_mlst:
            spaceName ='-'.join([root, space])
            catMembers[spaceName] = lst
            catMembers[root].append(spaceName)
    return catMembers


def create_training_tree(stem, trueTails, e2vec, knownDataDict=dict(),
                         mushroomStemRel="", embedSpace = "", enrichLevel=-1, enlargeMR = -1):
    tree = defaultdict(list)
    tree0 = defaultdict(list)
    vstem = ['*root*'] + stem
    for p, h in zip(vstem[:-1], vstem[1:]):
        tree[p] = [(mushroomStemRel, [h])]
        tree0[p] = [h]
    tree[vstem[-1]] = [(embedSpace, trueTails)]
    tree0[vstem[-1]] = trueTails
    if enrichLevel > 0:
        cstem = deepcopy(stem[:-1])
        for hi in cstem[-enrichLevel:]:
            canlst = tree[hi][0][1]
            for ele in knownDataDict[mushroomStemRel][hi]:
                if ele not in canlst:
                    canlst.append(ele)
            if enlargeMR > 0:
                eLst = [ele for ele in canlst[:enlargeMR]
                        if ele in e2vec and ele not in cstem[:-enrichLevel]  and ele != hi]
            else:
                eLst = [ele for ele in canlst if ele in e2vec
                        and ele not in cstem[:-enrichLevel] and ele != hi]
            tree[hi] = [(mushroomStemRel, eLst)]
            tree0[hi] = eLst
    return tree, tree0


def create_PLC(stem, trainingTree, posDic=None, width = 30, maxPLC = 300):

    fpDic = dict()
    if posDic:
        stem0 =['*root*'] + stem
        posDic.update({'*root*':1})
        for i, stemEle in enumerate(stem0):
            for space, leaves in trainingTree[stemEle]:
                for leaf in leaves:
                    fpDic[leaf] = [str(posDic.get(ele, 1)) for ele in stem0[:i+1]]
                    fpDic[leaf] += ['0'] * (width - len(fpDic[leaf]))
                    assert len(fpDic[leaf]) == width, leaf
    else:
        fpDic[stem[0]] = ['1'] + ['0'] * (width - 1)
        for i, e in enumerate(stem):
            for space, leaves in trainingTree[e]:
                for leaf in leaves:
                        fpDic[leaf] = ['1'] * (i+1+1) + ['0'] * (width - i -1 -1)
    return fpDic


def add_stem_hight_to_results(iResultFile='', iMushroomPath=''):
    newRlt = []
    if not os.path.isfile(iResultFile):
        print(iResultFile, ' not exist')
        return
    for ln in load_file_to_list(iResultFile):
        head, tail, rel, x, stemRel, y, mrV, v = ln.split()
        mushroomName = get_logfile_name(head, stemRel, rel)
        num = -1
        if not os.path.isfile(os.path.join(iMushroomPath, mushroomName)):
            print(os.path.join(iMushroomPath, mushroomName), ' not exist')
            continue
        for mLn in load_file_to_list(os.path.join(iMushroomPath, mushroomName)):
            if "-tr_contain" in mLn or "-ntr_contain" in mLn:
                num += 1
        newRlt.append(" ".join([ln, "SHeight=", str(num)]))
    with open(iResultFile+".STH", 'w') as ofh:
        ofh.write('\n'.join(newRlt)+'\n')
        ofh.flush()


def multi_parents(h):
    lst = h.split('-')
    if len(lst) == 1:
        return False
    for ele in lst:
        if '.' not in ele:
            return False
    return True


def get_truth_value_dynamic(h, t, r, relWNDic, ofile = ""):
        def get_values(sl, head, fun):
            pk = getattr(wn, sl)
            entity = pk(head)
            # print(head, fun)
            return [x.name() for x in getattr(entity, fun)()]

        A, B, rel = h, t, r
        synlem, func, inv = relWNDic[rel]

        if inv == "inv":
            flag = True
            h, t = B, A
        else:
            flag = False
            h, t = A, B
        if synlem == "lemma":
            h += "." + h.split('.')[0]

        vlst = get_values(synlem, h, func)
        if inv == "bi":
            vlst += get_values(synlem, t, func)

        if t in vlst and inv!="bi":
            value = "True"
            if flag:
                h, t = t, h
        elif inv == "bi" and h in vlst:
            value = "True"
        elif len(vlst) == 0:
            value = "unknown"
        else:
            value = "False"
        newline = ' '.join([h, t, rel, value])
        with open(ofile, 'a+') as ofh:
            ofh.write("\n"+newline)
        return value


def get_common_hypernyms(A, B, sl="synset", fun="common_hypernyms"):
        ws = getattr(wn, sl)
        entity = ws(A)
        # print(head, fun)
        return [x.name() for x in getattr(entity, fun)(ws(B))]


def get_hypernym_paths(A):
        ws = getattr(wn, "synset")
        entity = ws(A)
        llst = []
        for lst in getattr(entity, 'hypernym_paths')():
            llst.append([x.name() for x in lst])
        return llst


def enrich_stems_with_wordnet(A, cstem=[], fn_synsets=None):
    def wn_to_wn11_name(xname):
        if A.startswith('__'):
            nlst = xname.split('.')
            n0, num = nlst[0], nlst[2].lstrip('0')
            return "__"+n0+"_"+num

    def get_llst(B):
        llst0 = []
        if B.startswith('__'):
            word = '_'.join(B.split('_')[2:-1])
        else:
            return cstem

        for ws in fn_synsets(word):
            for lst in getattr(ws, 'hypernym_paths')():
                if wn_to_wn11_name(lst[-1].name()) == B:
                    llst0.append([wn_to_wn11_name(x.name()) for x in lst])
        return llst0

    llst = get_llst(A)
    if len(llst) == 0 and len(cstem)>1:
        A = cstem[-2]
        llst1 = get_llst(A)
        llst = [lx+[cstem[-1]] for lx in llst1]

    mstem, mlen = [], 0
    for alst in llst:
        sset = set(cstem).intersection(alst)
        if len(sset) > mlen:
            mlen = len(sset)
            mstem = alst
    if mstem:
        return mstem
    else:
        return cstem


def get_multi_parents_stem(h):

    cstem = []
    wslst = h.split('-')
    i, N = 0, len(wslst)
    for i in range(N):
        j = i + 1
        if j < N:
            cstem0 = get_common_hypernyms(wslst[i], wslst[j], sl="synset", fun="common_hypernyms")
            if len(cstem) == 0:
                cstem = cstem0
            else:
                cstem = [ele for ele in cstem if ele in cstem0]
    llst = get_hypernym_paths(wslst[0])
    for lst in llst:
        if set(lst) > set(cstem):
            return [e for e in lst if e in cstem]
    return cstem


def compute_vec_of_multi_parents(h, e2vec=dict()):
    v = [e2vec[ele] for ele in h.split('-')]
    return np.sum(np.array(v), axis=0)/len(v)


def mushroom_triple_classification(testTriples=[], knownDataDict=dict(), mushroomStemRel='',
                                   maxTrueTails=-1, ballMarginLst=[], widthOfPLC=30, plcDic=dict(), e2vec=dict(),
                                   eflag="", subspaceDim=20, addDim=[512] * 100, mPHypernymPathDic=dict(),
                                   enrichLevel=1, enlargeMR=-1, L0=decimal.Decimal(1e+100), R0=decimal.Decimal(1e-200),
                                   groundTruthDict=dict(), relNballWNFile="", mushroomPath="", logPath="",
                                   mainLog="", mushroomResultFile="", mPInterSec=False, enhanceStem=False,
                                   semaphorMr=threading.Semaphore(), semaphorRlt=threading.Semaphore()):
    """
    test triple (h, r, t) true or false
    trueTails are all true tails from the training+valid datasets, which form the cap of the mushroom
    trueAncestors of h are ancestors of h in training+valid datasets, which form the stem of the mushroom

    load entity2vec dictionary
    some components of the mushroom do not vectors, just remove them, and log them

    training nball of the mushroom, and save the transformation history into file, trans_history

    apply trans_history for construct the nball of t

    if nball of t is located inside of nball of h, return true, otherwise false

    :param testTriples:
    :param trainTripeFile:
    :param validTripleFile:
    :param mushroomStemRels:
    :param similarWordFile:
    :param flagOfUsingSimilarWord:
    :param widthOfPLC:
    :param entity2vecFile:
    :param ballMargin:
    :param addDim:
    :param logPath:
    :param resultPath:
    :return:
    """
    global entity2vecFile, word2vecFile, fn_synsets, MaxStemLength, DEBUG, recordEnhencedStem, enhencedStems

    def get_true_tails(h, r, edic=dict()):
        return [e for e in knownDataDict[r][h] if e in edic or e.split('.')[0] in edic]

    def get_mushroom_stem(hx, rel='', enhanceStem=False, edic=dict()):
        global fn_synsets

        def get_parent_pos(h0, parent):
            if len(h0.split('.')) == 2:
                pos = h0.split('.')[1]
                parentPos = list(filter(lambda ele: ele.split('.')[1] == pos, parent))
                return parentPos
            else:
                return []

        lst = [hx]
        i_rel = rel + "-1"
        if i_rel in knownDataDict:
            while True:
                parent = knownDataDict[i_rel].get(hx, False)
                if parent:
                    parentPos = get_parent_pos(h, parent)
                    if parentPos:
                        hx = parentPos[0]
                    else:
                        hx = parent[0]
                    if hx not in lst:
                        lst.append(hx)
                    else:
                        break
                else:
                    break
        lst.reverse()
        lst = [e for e in lst if e in edic or e.split('.')[0] in edic]
        if enhanceStem and len(lst) < MaxStemLength:
            lst = enrich_stems_with_wordnet(h, cstem=lst, fn_synsets=fn_synsets)
        return [e for e in lst if e in edic or e.split('.')[0] in edic]

    def create_cat_path(stem, leaves, tree=None):
        cpDic = dict()
        if tree:
            nodeLst = ['*root*']
            cpDic['*root*'] = []
            while nodeLst:
                nd = nodeLst.pop()
                children = tree.get(nd, [])
                if children:
                    nodeLst += children
                for child in children:
                    cpDic[child] = cpDic[nd] + [nd]
        else:
            for leaf in leaves:
                cpDic[leaf] = stem
            for i in range(len(stem)):
                cpDic[stem[i]] = stem[:i]
        return cpDic

    def load_nball_wn_relation(relfile):
        funDic = dict()
        for relMapping in load_file_to_list(relfile):
            wlst = relMapping.split()
            if len(wlst) > 2:
                nballrel, wnrel = wlst[:2]
                funDic[wlst[0]] = wnrel.split('.')
        return funDic

    def exist_mushroom(logFileName, outputPath=""):
        return os.path.exists(os.path.join(outputPath, logFileName))

    def get_result_from_one_mushroom(h, ballMargin, mushroomStemRel, r):
        posixMgEvec = str(ballMargin) + eflag
        if enlargeMR > 0:
            posixMgEvec += "_enlarge" + str(enlargeMR)
        mushroomPath0 = mushroomPath + posixMgEvec
        semaphorMr.acquire()
        if not os.path.exists(mushroomPath0):
            os.makedirs(mushroomPath0)
        semaphorMr.release()

        logPath0 = logPath + posixMgEvec
        semaphorMr.acquire()
        if not os.path.exists(logPath0):
            os.makedirs(logPath0)
        semaphorMr.release()

        mushroomName = get_logfile_name(h, mushroomStemRel, r)
        logFile0 = os.path.join(logPath0, mushroomName)
        mushroomResultFile0 = mushroomResultFile + posixMgEvec

        posixEvecMargin1 = '1.0' + eflag
        logPathMargin1 = logPath + posixEvecMargin1
        mushroomPathMargin1 = mushroomPath + posixEvecMargin1
        mushroomNameMargin1 = os.path.join(logPathMargin1, mushroomName)
        if ballMargin != 1:
            ballDict = load_entity_vector(os.path.join(mushroomPathMargin1, mushroomName))
            ballDict = update_ball_margins(ballDict, ballMargin)
        elif not DEBUG and exist_mushroom(mushroomName, outputPath=mushroomPath0):
            ballDict = load_entity_vector(os.path.join(mushroomPath0, mushroomName))
        else:
            ballDict = training_one_family_multi_relations(treeStruc=trainingTree,
                                                           root=trainingTree['*root*'][0][1][0],
                                                           w2vDic=e2vec, catMemDic=catMemDic,
                                                           catFPDic=catFPDic, ballDict=dict(),
                                                           subspaceDim=subspaceDim,
                                                           L0=L0, R0=R0, addDim=addDim, logFile=logFile0)

            lh = open(logFile0, 'a')
            lh.flush()
            lh.close()
            semaphorMr.acquire()
            create_ball_files(ballDict, mushroomName, outputPath=mushroomPath0)
            semaphorMr.release()

        tball = create_testing_ball(t, h, subspace=r, subspaceDim=subspaceDim, code=tPLC, w2vDic=e2vec,
                                    catPathDic=cpDic, catFPDic=catFPDic, addDim=addDim, L0=L0, R0=R0,
                                    logFile=mushroomNameMargin1)
        return ballDict['-'.join([h, r])], ballDict[h], tball, mushroomResultFile0

    mushroomResults = defaultdict(list)
    mainLog += eflag
    open(mainLog, 'a+').close()

    # relWNDic = load_nball_wn_relation(relNballWNFile)

    testLst = testTriples

    for ln in testLst:
        print('TC processing ', ln)
        h, t, r, v = ln.split()

        if not has_e2v(t, e2vec):
            # print('False:', ln)
            semaphorMr.acquire()
            with open(mainLog, 'a') as ofh:
                ofh.write(' '.join([t, 'is not in entity vec\n']))
            semaphorMr.release()
            continue
        allTrueTails = get_true_tails(h, r, edic=e2vec)
        if t in allTrueTails:
            allTrueTails.remove(t)

        if len(allTrueTails) == 0:
            with open(mainLog, 'a') as ofh:
                semaphorMr.acquire()
                ofh.write(' '.join([h, r, 'has no tails in KG\n']))
                semaphorMr.release()
            continue
        elif maxTrueTails > 0:
            allTrueTails.sort()
            trueTails = allTrueTails[:maxTrueTails]
        else:
            trueTails = allTrueTails

        if not multi_parents(h):
            stem = get_mushroom_stem(h, rel=mushroomStemRel, enhanceStem=enhanceStem, edic=e2vec)
            if recordEnhencedStem:
                for tail in trueTails:
                    enhencedStems.append(' '.join(stem + [tail]))
                continue

        elif mPInterSec:
            stems = []
            for h0 in h.split('-'):
                # get from WN
                stem = mPHypernymPathDic.get(h0, False)
                if not stem:
                    stem = [ele for ele in get_hypernym_paths(h0)[0] if ele in e2vec]  # to improve
                    mPHypernymPathDic[h0] = stem
                stems.append(stem)
        else:
            stem = get_multi_parents_stem(h)
            stem += [h]
            e2vec[h] = compute_vec_of_multi_parents(h, e2vec=e2vec)

        if not mPInterSec:
            stemHeight = len(stem) - 1
            if len(stem) == 0 or t in stem or len(stem + trueTails) != len(set(stem + trueTails)):
                # print('False:', ln)
                with open(mainLog, 'a') as ofh:
                    semaphorMr.acquire()
                    ofh.write(' '.join(stem + trueTails + ['has no tails in KG\n']))
                    semaphorMr.release()
                continue
            h = stem[-1]

            trainingTree, trainingTree0 = create_training_tree(stem, trueTails, e2vec, knownDataDict=knownDataDict,
                                                               mushroomStemRel=mushroomStemRel, embedSpace=r,
                                                               enrichLevel=enrichLevel, enlargeMR=enlargeMR)

            catMemDic = make_cat_members_from_multi_space_dic(trainingTree)
            catFPDic = create_PLC(stem, trainingTree, posDic=plcDic, width=widthOfPLC)

            tPLC = catFPDic[trueTails[0]]
            cpDic = create_cat_path(stem, trueTails, tree=trainingTree0)
            # print(ln, cpDic)
            if not cpDic:
                # print('False:', ln)
                semaphorMr.acquire()
                with open(mainLog, 'a') as ofh:
                    ofh.write(' '.join([h] + ['has no stems in KG\n']))
                semaphorMr.release()
                continue
            if DEBUG: ballMarginLst = ballMarginLst[:1]
            for ballMargin in ballMarginLst:
                hballsub, hball, tball, mushroomResultF0 = get_result_from_one_mushroom(h, ballMargin, mushroomStemRel,
                                                                                        r)
                mrValueSubspace = qsr_P(tball, hballsub)
                mrValue = qsr_P(tball, hball)
                if mrValueSubspace:  # qsr_P(tball, ballDict['-'.join([h, r])]):
                    if v in ["True", "1"]:
                        recordLine = ' '.join([h, t, r, "stem=", mushroomStemRel, str(len(stem)), "mrT", "1"])
                    elif v in ["False", "-1"]:
                        recordLine = ' '.join([h, t, r, "stem=", mushroomStemRel, str(len(stem)), "mrT", "-1"])
                    else:
                        recordLine = ' '.join([h, t, r, "stem=", mushroomStemRel, str(len(stem)), "mrT", "0"])
                elif mrValue:  # qsr_P(tball, ballDict[h]) and not multi_parents(h):
                    if v in ["True", "1"]:
                        recordLine = ' '.join([h, t, r, "stem=", mushroomStemRel, str(len(stem)), "mrTx", "1"])
                    elif v in ["False", "-1"]:
                        recordLine = ' '.join([h, t, r, "stem=", mushroomStemRel, str(len(stem)), "mrTx", "-1"])
                    else:
                        recordLine = ' '.join([h, t, r, "stem=", mushroomStemRel, str(len(stem)), "mrTx", "0"])
                else:
                    if v in ["True", "1"]:
                        recordLine = ' '.join([h, t, r, "stem=", mushroomStemRel, str(len(stem)), "mrF", "1"])
                    elif v in ["False", "-1"]:
                        recordLine = ' '.join([h, t, r, "stem=", mushroomStemRel, str(len(stem)), "mrF", "-1"])
                    else:
                        recordLine = ' '.join([h, t, r, "stem=", mushroomStemRel, str(len(stem)), "mrF", "0"])
                if ballMargin == 1.0:
                    print(recordLine)
                mushroomResults[mushroomResultF0].append(" ".join([recordLine, "SHeight=", str(stemHeight)]))


        semaphorRlt.acquire()
        for rfile, vlst in mushroomResults.items():
            with open(rfile, 'w') as ofh:
                ofh.write('\n'.join(vlst) + '\n')
        semaphorRlt.release()

def pos_match(ele, lst):
    pos1 = ele.split('.')[1]
    if pos1 in [ele.split('.')[1] for ele in lst]:
        return True
    else:
        return False


def mushroom_triple_classification_with_different_pars(threadNum = 3,
                                                       testTripleFile="" , trainTripeFile="", validTripleFile="",
                                                       mushroomStemRel="", maxTrueTails=-1, enhanceStem=False,
                                                       enrichLevel=-1, enlargeMR = -1, multiParentDicFile ='',
                                                       widthOfPLC=0, plcFile=None, e2vecFile="", ballMarginLst=[],
                                                       subspaceDim = 20, addDim=[], mPInterSec = False,
                                                       L0=0, R0=0, mushroomGroundTruth="", relNballWNFile="",
                                                       mushroomPath="", logPath="", mainLog = "",
                                                       mushroomResultFile=""):

    class myThread(threading.Thread):

        def __init__(self, i, semaphorMr, semaphorRlt, e2vecFile):
            threading.Thread.__init__(self)
            self.semMr = semaphorMr
            self.semRlt = semaphorRlt
            self.startIndex = i * unit
            self.endIndex = min((i+1)*unit, len(allTestTriples))

        def run(self):
            mushroom_triple_classification(testTriples=allTestTriples[self.startIndex:self.endIndex],
                                           knownDataDict=knownDataDict, mushroomStemRel=mushroomStemRel,
                                            maxTrueTails=maxTrueTails, enrichLevel=enrichLevel, enlargeMR=enlargeMR,
                                            widthOfPLC=widthOfPLC, plcDic=plcDic, e2vec=e2vec, eflag = eflag,
                                            ballMarginLst=ballMarginLst,
                                            subspaceDim=subspaceDim, enhanceStem=enhanceStem,
                                            mPHypernymPathDic = mPHypernymPathDic,
                                            addDim=addDim, L0=L0, R0=R0, groundTruthDict=groundTruthDict,
                                            relNballWNFile=relNballWNFile, mPInterSec = mPInterSec,
                                            mushroomPath=mushroomPath, logPath=logPath, mainLog=mainLog,
                                            mushroomResultFile=mushroomResultFile,
                                            semaphorMr=self.semMr, semaphorRlt=self.semRlt)

    allTestTriples = load_file_to_list(testTripleFile)
    unit = math.ceil(len(allTestTriples) / threadNum)
    semaphorMr = threading.Semaphore()
    semaphorRlt = threading.Semaphore()

    print('e2vecFile', e2vecFile)
    e2vec = load_entity_vector(e2vecFile)
    eflag = get_symbol_dic_for_vec_file(e2vecFile)
    knownDataDict = load_train_dataset([trainTripeFile, validTripleFile])

    mPHypernymPathDic = defaultdict()
    if os.path.isfile(multiParentDicFile):
        for lst in load_file_to_list(multiParentDicFile):
            elst = lst.split()
            mPHypernymPathDic[elst[0]] = elst[1:]

    groundTruthDict = dict()
    for lst in load_file_to_list(mushroomGroundTruth):
        elst = lst.split()
        groundTruthDict[' '.join(elst[:-1])] = elst[-1]

    plcDic = dict()
    if plcFile:
        for lst in load_file_to_list(plcFile):
            pos = lst.split()
            plcDic[pos[0]] = int(pos[1]) + 1

    for i in range(threadNum):
        thread = myThread(i, semaphorMr, semaphorRlt, e2vecFile)
        thread.start()


def get_symbol_dic_for_vec_file(vfile):
    if "TEKEE" in vfile:
        return "TE"
    if "TEKEH" in vfile:
        return "TH"
    if "TransE" in vfile:
        return "TransE"











