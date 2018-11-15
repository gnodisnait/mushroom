import os
import pprint
import matplotlib.pyplot as plt
from copy import deepcopy
from collections import defaultdict
from mushroom.config import e2vFlags, defaultGammaLst, e2vDic


def is_positive(x):
    if x in ["mrT"]:
        return True
    else:
        return False


def is_ground_truth(x):
    if x in ["1"]:
        return True
    else:
        return False


def print_latex_tables(dic, e2vLst, gammaLst, result):
    lns = []
    lns.append('|'.join(['c'] * (len(gammaLst)+1)))
    lns.append('&'.join(['']+[str(e) for e in gammaLst]))
    for e2vFlag in e2vLst:
        values = dic[e2vFlag]
        lns.append('&'.join([e2vFlag]+[str("{:.2f}".format(e[result]) ) for e in values]))
        print(' \\\hline '.join(lns))


def get_accuracy_precision_recall_from_series_with_stem_length(dataSet = "", gammaLst=defaultGammaLst,
                                                               e2vLst=e2vFlags,
                                                               result='precision', stemHeight=-1, legendloc=''):
    dic = defaultdict(list)
    lst = deepcopy(stemHeight)
    for e2vFlag in e2vLst:
        while stemHeight:
            maxValue = 0
            for gamma in [str(ele) for ele in gammaLst]:
                filePat = dataSet + "{}" + "{}"
                ifile = filePat.format(gamma, e2vFlag)
                dic0=get_accuracy_precision_recall(ifile, stemHeight=stemHeight[0])
                if dic0[result] == 0:
                    break
                if maxValue == 0:
                    maxValue = [gamma, dic0[result], stemHeight[0]]
                elif maxValue[1]< dic0[result]:
                    maxValue = [gamma, dic0[result], stemHeight[0]]
            if maxValue !=0:
                    dic[e2vFlag].append(maxValue)
            stemHeight = stemHeight[1:]
        stemHeight = deepcopy(lst)

    colors = ['b', 'r', 'g', 'y']
    marks = ['o', 'X', '<']
    ll = []
    i = 0
    for key in e2vLst:
        c = colors[i]
        m = marks[i]
        X = [ele[2] for ele in dic[key]]
        Y = [ele[1] for ele in dic[key]]
        G = [ele[0] for ele in dic[key]]
        l1, = plt.plot(X, Y, marker=m, linestyle='-', color=c)
        for xy in zip(X,Y):
            x = xy[0]
            id = X.index(x)
            g = G[id]
            plt.annotate('%s' %g, xy=xy, textcoords='data', color=c)
        i += 1
        ll.append(l1)

    plt.legend(ll, [e2vDic[ele] for ele in e2vLst], loc=legendloc)
    plt.ylabel(result)
    plt.show()


def get_accuracy_precision_recall_from_series(dataSet = "", gammaLst=[], e2vLst=e2vFlags,
                                              result='precision', stemHeight=-1, legendloc=''):
    dic = defaultdict(list)
    for e2vFlag in e2vLst:
        for gamma in gammaLst:
            filePat = dataSet+"{}"+"{}"
            ifile = filePat.format(gamma, e2vFlag)
            dic0 = get_accuracy_precision_recall(ifile, stemHeight=stemHeight)
            dic[e2vFlag].append(dic0)

    print_latex_tables(dic, e2vLst, gammaLst, result)

    colors = ['b', 'r', 'g', 'y']
    marks = ['o', 'X', '<']
    ll = []
    i = 0
    for key in e2vLst:
        c = colors[i]
        m = marks[i]
        l1, = plt.plot([float(ele) for ele in gammaLst],  [ele[result] for ele in dic[key]], marker=m,
                                   linestyle='-', color=c)
        i += 1
        ll.append(l1)

    plt.legend(ll, [e2vDic[ele] for ele in e2vLst], loc=legendloc)
    plt.ylabel(result)
    plt.show()


def get_accuracy_precision_recall(ifile="",  relation='', stemHeight=-1):
    "settle.v.03 scatter.n.02 tr_contain1 stem= tr_contain1 3 mrF -1 SHeight= 2"
    def is_relation(rel0, relation0=''):
        if len(relation0) == 0 or rel0 == relation0:
            return True
        else:
            return False

    def is_stem_height(height0, stemHeight0=''):
        if stemHeight0 == -1 or height0 == stemHeight0:
            return True
        elif type(stemHeight0) == list and str(height0) in stemHeight0:
            return True
        else:
            return False

    with open(ifile, 'r') as ifh:
        tp, fp, fn, tn = 0, 0, 0, 0
        for ln in ifh:
            wlst = ln[:-1].split()
            rel, mrResult, groundTruth, height = wlst[2], wlst[6], wlst[7], int(wlst[5])
            if is_relation(rel, relation0=relation) and is_stem_height(height, stemHeight0=stemHeight):
                if is_positive(mrResult) and is_ground_truth(groundTruth):
                    tp += 1
                elif is_positive(mrResult) and not is_ground_truth(groundTruth):
                    fp += 1
                elif not is_positive(mrResult) and is_ground_truth(groundTruth):
                    tn += 1
                elif not is_positive(mrResult) and not is_ground_truth(groundTruth):
                    fn += 1
    if tp + fp > 0:
        precision = tp/(tp + fp)
    else:
        precision = -1
    if tp + tn > 0:
        recall = tp/(tp + tn)
    else:
        recall = -1
    if tp + fn + fp + tn == 0:
        accuracy = 0
    else:
        accuracy = (tp + fn)/(tp + fn + fp + tn)
    return {'accuracy': accuracy,
            'precision': precision,
            'recall': recall}


def get_max_from_series(ipath="", filePat="", dataSets = [], gammaLst=[], e2vLst=[],
                                              result='precision', stemLst=[]):
    dic = defaultdict()
    for dataSet in dataSets:
        for e2vFlag in e2vLst:
            for height in stemLst:
                mxv = 0
                for gamma in gammaLst:
                    fpath = ipath.format(dataSet)
                    tcFile = filePat.format(gamma, e2vFlag)
                    ifile = os.path.join(fpath, tcFile)
                    dic0 = get_accuracy_precision_recall(ifile, stemHeight=int(height))
                    if mxv < dic0[result]:
                        mxv = dic0[result]
                        dic[height] = mxv
    pprint.pprint(dic)
    return dic


def analysis_with_different_length_stems(dataSet="", gammaLst=defaultGammaLst, e2vLst=e2vFlags,
                                         target='precision', stemLst=[], plotGamma = 1.0,legendloc='lower left'):
    dic = defaultdict()
    plotDic = {}
    for e2vFlag in e2vLst:
        dic[e2vFlag] = defaultdict()
        for gamma in gammaLst:
            dic[e2vFlag][gamma] = defaultdict()
            for height in stemLst:
                filePat = dataSet + "{}" + "{}"
                ifile = filePat.format(gamma, e2vFlag)

                dic[e2vFlag][gamma][height] = get_accuracy_precision_recall(ifile, stemHeight=int(height))[target]
            if gamma == plotGamma:
                dic0 = {}
                for h, v in dic[e2vFlag][gamma].items():
                    if v!=-1:
                        dic0[h] = v
                plotDic[e2vFlag] = dic0

    vis_2D(plotDic,  e2vLst=e2vLst, result = target, legendloc=legendloc)
    pprint.pprint(plotDic)
    return dic


def vis_2D(pdic, e2vLst=[], result = 'precision', xlable='lengths of type chains', legendloc='center right'):
    colors = ['b', 'r', 'g', 'y']
    marks = ['o', 'X', '<']
    ll = []
    i = 0
    legends = []
    for dname in e2vLst:
        c = colors[i]
        m = marks[i]
        legends.append(e2vDic[dname])
        print(pdic)
        dic = {float(k): v for (k, v) in pdic[dname].items()}
        dkeys = list(dic.keys())
        dkeys.sort()
        l1, = plt.plot(dkeys, [dic[k] for k in dkeys],  marker=m, linestyle='-', color=c)
        i += 1
        ll.append(l1)
    plt.legend(ll, legends, loc=legendloc)
    plt.ylabel(result)
    plt.xlabel(xlable)
    plt.show()
