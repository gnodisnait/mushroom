import os
import numpy as np
import decimal
from io import BytesIO
from zipfile import ZipFile
import urllib.request
from collections import defaultdict
from mushroom.config import Precision, NBallLinks, KGS

decimal.getcontext().prec = Precision


def download_nball_files(dataPath=""):
    if "all" in KGS:
        KGS.remove("all")
    for kg in KGS:
        url = urllib.request.urlopen(NBallLinks[kg])
        with ZipFile(BytesIO(url.read())) as zipFiles:
            for afile in zipFiles.namelist():
                location = os.path.join(dataPath, kg)
                if not os.path.exists(location):
                    os.makedirs(location)
                file = zipFiles.open(afile)
                content = file.read()
                with open(os.path.join(location, afile), 'wb') as ofh:
                    ofh.write(content)


def load_file_to_list(ifile):
    lst = []
    with open(ifile, 'r') as ifh:
        for ln in ifh:
            lst.append(ln[:-1])
    return lst


def load_entity_vector(entity2vecFile):
    e2vlst = load_file_to_list(entity2vecFile)
    e2vDic = dict()
    for e2v in e2vlst:
        elst = e2v.split()
        e2vDic[elst[0]] = [decimal.Decimal(ele) for ele in elst[1:]]
    return e2vDic


def create_testing_ball(bname, cat, subspace=',', subspaceDim=0,
                        code=None, w2vDic=None, catPathDic=None, catFPDic=None, addDim=[], L0=0, R0=0,
                        logFile=""):
    if code:
        fingerPrintUpCat = code
    else:
        fingerPrintUpCat = catFPDic[bname]

    if bname in w2vDic:
        wvec = w2vDic[bname]
    else:
        word = bname.split('.')[0]
        wvec = w2vDic[word]
    subspaceCode = get_subspace_code(subspace, subspaceDim)
    w2v = [decimal.Decimal(ele * 100) for ele in wvec]
    cpoint = w2v + [decimal.Decimal(ele) + 10 for ele in fingerPrintUpCat] + subspaceCode + addDim
    tball = vec_norm(cpoint) + [L0, R0]
    tball = shitfing_htrans_one_testing_ball(tball, cat, catPathDic=catPathDic, logFile=logFile)
    return tball


def shitfing_htrans_one_testing_ball(tball, parentName, catPathDic=None, logFile=""):
    """
    suppose parent is the parent node of the testChild, shift testChild according to the shifting history of parent
    :param tball :
    :param parent :
    :return:
    """
    ansLst = [parentName] + catPathDic[parentName]
    # cloc = copy.deepcopy(tball)
    transh = get_trans_history(logFile)
    for trans in transh:
        if trans[0] == 's' and trans[1] in ansLst:
            cvec, cL, cR = tball[:-2], tball[-2], tball[-1]
            vec, L = trans[2][:-1], trans[2][-1]
            val = cL * cL + L * L + 2 * cL * L * vec_cos(cvec, vec)
            # l = np.sqrt(val)
            npoint = vec_point(cvec, cL) + vec_point(vec, L)
            newVec = vec_norm(npoint)
            l1 = np.linalg.norm(npoint)
            tball[:-1] = newVec+ [l1]

        if trans[0] == 'h' and trans[1] in ansLst:
            ratio = trans[2]
            tball[-2] *= ratio
            tball[-1] *= ratio

        if transh[0] == 'r' and trans[1] in ansLst:
            rotatedCos = vec_cos(trans[2], trans[3])
            vecNew = rotate(tball, rotatedCos)
            tball = vecNew + tball[-2:]
    return tball


def get_trans_history(logFile = None):
    """
    :param sFile:
    :return:
    """
    transh = []
    with open(logFile, 'r') as ifh:
        for ln in ifh:
            wlst = ln[:-1].split()
            if wlst[0] == "shifting":
                transh.append(['s', wlst[1], [decimal.Decimal(ele) for ele in wlst[2:]]])
            elif wlst[0] == "homo":
                transh.append(['h', wlst[1], decimal.Decimal(wlst[2])])
            elif wlst[0] == "rotate":
                rline = " ".join(wlst[2:])
                fromln, toln = rline.split('TO')
                transh.append(['r',
                               wlst[1],
                               [decimal.Decimal(ele) for ele in fromln.split()],
                               [decimal.Decimal(ele) for ele in toln.split()]])
    return transh


def update_ball_margins(ballDict, ballMargin):
    for key, val in ballDict.items():
        ballDict[key][-1] *= decimal.Decimal(ballMargin)
    return ballDict


def load_subspace_dict():
    groupLen = 4
    subSpaceDict = {"ntr_contain0": [5]*groupLen,
                    "ntr_contain1": [0]*groupLen*3 + [5]*groupLen,
                    "ntr_contain2": [0]*6 + [5]*groupLen,
                    "ntr_contain3": [0]*14 + [5]*groupLen,
                    "tr_contain0": [0] * groupLen + [9]*groupLen,
                    "tr_contain1": [0],
                    "tr_contain2": [0] * groupLen*2 + [9]*groupLen,
                    "sym_similar": [2]*groupLen + [0]*groupLen + [2]*groupLen,
                    "sym_also_see": [0]*groupLen+[2]*groupLen + [0]*groupLen + [2]*groupLen,
                    "sym_related": [0]*5+[6]*5 + [0]*5 + [6]*5,
                    "sym_spouse": [1]*5+[0]*5 + [0]*5 + [9]*5,
                    "ntr_contain13": [1,2,3,4,5]+[-1,-2,-3,-4,-5] + [1,2,3,4,5]+[-1,-2,-3,-4,-5],
                    "ntr_contain14": [-1,-2,-3,-4,-5] + [1,2,3,4,5]+ [1,2,3,4,5]+[-1,-2,-3,-4,-5],
                    "ntr_contain18": [0,-2,-3,-4,0] + [0,2,3,4,0]+ [0,2,3,4,0]+[0,-2,-3,-4,0],
                    "ntr_contain19": [-1,0,-3,-4,-5] + [1,0,3,4,5]+ [1,0,3,4,5]+[-1,0,-3,-4,-5],
                    "ntr_contain20": [-1,-2,0,-4,-5] + [1,2,0,4,5]+ [1,2,0,4,5]+[-1,-2,0,-4,-5],
                    "ntr_contain21": [-1,-2,-3,0,-5] + [1,2,3,0,5]+ [1,2,3,0,5]+[-1,-2,-3,0,-5],
                    "ntr_contain22": [-1,-2,-3,-4,0] + [1,2,3,4,0]+ [1,2,3,4,0]+[-1,-2,-3,-4,0],
                    "ntr_contain23": [0,0,-3,-4,-5] + [0,0,3,4,5]+ [0,0,3,4,5]+[0,0,-3,-4,-5],
                    "ntr_contain24": [-1,-2,0,0,-5] + [1,2,0,0,5]+ [1,2,0,0,5]+[-1,-2,-3,0,-5],
                    "ntr_contain25": [-1,-2,-3,0,0] + [1,2,3,0,0]+ [1,2,3,0,0]+[-1,-2,-3,0,0],
                    "ntr_contain26": [0,-2,-3,-4,0] + [0,2,3,4,0]+ [0,2,3,4,0]+[0,-2,-3,-4,0],
                    "ntr_contain27": [0,-2,0,-4,0] + [0,2,0,4,0]+ [0,2,0,4,0]+[0,-2,0,-4,0],
                    }
    return subSpaceDict


def load_train_dataset(tripleFiles):
    kdic = defaultdict()
    for tfile in tripleFiles:
        with open(tfile, 'r') as ifh:
            for ln in ifh:
                if '__day_5' in ln:
                    print('--')
                A, B, rel = ln[:-1].split()
                subDict = kdic.get(rel, 'not-exist')
                if subDict == 'not-exist':
                    subDict = defaultdict(list)
                    kdic[rel] = subDict
                if B not in subDict[A]:
                    subDict[A].append(B)

                i_rel = rel+"-1"
                subDict = kdic.get(i_rel, 'not-exist')
                if subDict == 'not-exist':
                    subDict = defaultdict(list)
                    kdic[i_rel] = subDict
                if A not in subDict[B]:
                    subDict[B].append(A)
        if 'sym_related' not in kdic:
            kdic['sym_related'] = defaultdict()
    return kdic


def get_subspace_code(subspacename = '', spacespaceDim = 20):
    dic = load_subspace_dict()
    code = dic.get(subspacename, [0])
    code += [0] * (spacespaceDim - len(code))
    return code


def vec_norm(v):
    decimal.getcontext().prec = 400
    l = decimal.Decimal(np.linalg.norm(v))
    return [decimal.Decimal(ele)/l for ele in v]


def ball_norm(ballV):
    cpoint = vec_norm(ballV[:-2])
    return cpoint + ballV[-2:]


def vec_point(v, l):
    """
    :param v: v is the unit vector
    :param l: l is the lenth
    :return:
    """
    v1 = [decimal.Decimal(ele) for ele in v]
    l1 = decimal.Decimal(l)
    return np.multiply(v1, l1)


def vec_cos(v1, v2):
    v1 = [decimal.Decimal(ele) for ele in v1]
    v2 = [decimal.Decimal(ele) for ele in v2]
    return np.dot(v1, v2)


def average_vector(vecLst):
    N = len(vecLst)
    return np.divide(np.sum(vecLst, axis=0), N)


def dis_between_norm_vec(vec1, vec2):
    cos = np.dot([decimal.Decimal(ele) for ele in vec1], [decimal.Decimal(ele) for ele in vec2])
    d2 = 2 - 2 * decimal.Decimal(cos)
    # v1 = np.multiply([decimal.Decimal(ele) for ele in ball1[:-2]], ball1[-2])
    # v2 = np.multiply([decimal.Decimal(ele) for ele in ball2[:-2]], ball2[-2])
    # return dis_between(v1, v2)
    if d2 <0:
        return 0
    return np.sqrt(d2)


def dis_between_ball_centers(ball1, ball2):
    if ball1 == ball2:
        return 0
    cos = np.dot([decimal.Decimal(ele) for ele in ball1[:-2]], [decimal.Decimal(ele) for ele in ball2[:-2]])
    d2 = decimal.Decimal(ball1[-2] * ball1[-2] + ball2[-2] * ball2[-2]) \
         - 2 * decimal.Decimal(ball1[-2]) * decimal.Decimal(ball2[-2]) * decimal.Decimal(cos)
    if d2 <0:
        return 0
    return np.sqrt(d2)


def qsr_P_degree(ball1, ball2):
    dis = dis_between_ball_centers(ball1, ball2)
    return ball2[-1] - dis - ball1[-1]


def qsr_P(ball1, ball2):
    degree = qsr_P_degree(ball1, ball2)
    if degree < 0:
        return False
    return True


def qsr_DC(ball1, ball2):
    """
        check whether ball1 disconnects from ball2
        :return: boolean
    """
    degree = qsr_DC_degree(ball1, ball2)
    if degree < 0:
        return False
    return True


def qsr_DC_degree(ball1, ball2):
    """
        check whether ball1 disconnects from ball2
        :return: boolean
    """
    dis = dis_between_ball_centers(ball1, ball2)
    return dis - ball1[-1]-ball2[-1]


def rotate(vec, cosine):
    """

    :param vec:
    :param alpha:
    :return:
    """
    i = 100
    while cosine >= 1:
        cosine = 1 - abs(decimal.Decimal('-1e-'+str(i)))
        i -= 1

    while True:
        sinV = 1 - cosine*cosine
        if sinV < 0:
            sinV = 0
        else:
            sinV = np.sqrt(sinV)
        i = -1
        while vec[i] == 0:
            i -= 1
        j = i - 1
        while vec[j] == 0:
            j -= 1

        vecI0, vecJ0 = vec[i], vec[j]

        vec[i] = cosine *vec[i] + sinV*vec[j]
        vec[j] = -sinV *vec[i] + cosine*vec[j]

        if vec[i] == vecI0 and vec[j] == vecJ0:
            i -= 1
            cosine = 1 - abs(decimal.Decimal('-1e-'+str(i)))
        else:
            break
    return vec
