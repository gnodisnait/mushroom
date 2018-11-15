import argparse
import os
from mushroom.config import defaultGammaLst
from mushroom import get_accuracy_precision_recall_from_series
from mushroom import get_accuracy_precision_recall_from_series_with_stem_length
from mushroom import analysis_with_different_length_stems
from mushroom import mushroom_triple_classification_with_different_pars
from mushroom.config import tekee2vecFiles, tekeh2vecFiles,transe2vecFiles, mushroomStrucFiles
from mushroom.config import mushroomPaths, mushroomTrainFiles, mushroomValidFiles, mushroomTestFiles
from mushroom.config import mushroomLogPaths, mushroomLogFiles, mushroomTCResults, relNballWNFile
from mushroom.config import ballMarginLst, dataPath, resultDir, KGS, e2vFlags
from mushroom.config import defaultStemLst, defaultEnhanceStem, defaultEnlargeMR, defaultL0
from mushroom.config import defaultR0, defaultAddDim, defaultEnrichLevel, defaultMaxTrueTails, defaultMPInterSec
from mushroom.config import defaultStemRel, defaultSubspaceDim, defaultWidthOfPLC

"""
# usage 0: download datasets
$ python mushroom.py --func download 

# usage 1: triple classification 
$ python mushroom.py --func tc --kg Wordnet18|Wordnet11|Freebase13|all --e2v TH|TE|TransE|all

# usage 2: visualize precision - gamma, recall - gamma, accuracy - gamma
$ python mushroom.py --vis_gamma recall --kg Wordnet18  
                     
# usage 3: visualize precision - length-of-type-chain
$ python mushroom.py --vis_length precision --kg Wordnet18    

# usage 4: visualize max_accuracy - length-of-type-chain, which-gamma
$ python mushroom.py --vis_mg accuracy --kg Wordnet18  --legendLoc lower right                      
"""


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--vis_gamma')
    parser.add_argument('--gammaList', type=float, nargs='*',
                        default=defaultGammaLst)
    parser.add_argument('--stemList', type=int, nargs='*',
                        default=defaultStemLst)
    parser.add_argument('--legendLoc', nargs='*', default=['upper', 'right'])

    parser.add_argument('--vis_length')
    parser.add_argument('--vis_mg')

    parser.add_argument('--kg')
    parser.add_argument('--e2v')

    parser.add_argument('--func')

    args = parser.parse_args()

    """
    $ python mushroom.py --vis_gamma recall --kg Wordnet18  --legendLoc center right
    """
    if args.vis_gamma and args.kg and args.gammaList and args.legendLoc:
        dataPattern = os.path.join(dataPath, args.kg, resultDir, "MRTCresult")
        if 'all' in e2vFlags:
            e2vFlags.remove('all')
        if 'all' in KGS:
            KGS.remove('all')
        get_accuracy_precision_recall_from_series(dataSet=dataPattern,
                                                  gammaLst=args.gammaList,
                                                  result=args.vis_gamma,
                                                  legendloc=" ".join(args.legendLoc))

    """
    $ python mushroom.py --vis_length accuracy --kg Wordnet18 --legendLoc lower right
    """
    if args.vis_length and args.kg and args.stemList and args.legendLoc:
        dataPattern = os.path.join(dataPath, args.kg, resultDir, "MRTCresult")
        if 'all' in e2vFlags:
            e2vFlags.remove('all')
        if 'all' in KGS:
            KGS.remove('all')
        analysis_with_different_length_stems(dataSet=dataPattern,
                                             target=args.vis_length,
                                             stemLst=args.stemList,
                                             legendloc=" ".join(args.legendLoc))

    """
    $ python mushroom.py --vis_mg accuracy --kg Wordnet18 --legendLoc lower right 
    """
    if args.vis_mg and args.kg and args.stemList and args.legendLoc:
        dataPattern = os.path.join(dataPath, args.kg, resultDir, "MRTCresult")
        if 'all' in e2vFlags:
            e2vFlags.remove('all')
        if 'all' in KGS:
            KGS.remove('all')
        get_accuracy_precision_recall_from_series_with_stem_length(dataSet=dataPattern,
                                                                   result=args.vis_mg,
                                                                   stemHeight=args.stemList,
                                                                   legendloc=" ".join(args.legendLoc))

    """
    $ python mushroom.py --func tc --kg Wordnet18 --e2v TH 
    """
    if args.func == "tc" and args.kg in KGS and args.e2v in e2vFlags:
        def process_one_kg(oneKg=None, entityEmbeddings=[]):
            mushroomPath = mushroomPaths[oneKg]
            mushroomStrucFile = mushroomStrucFiles[oneKg]

            mushroomTrainFile = mushroomTrainFiles[oneKg]
            mushroomValidFile = mushroomValidFiles[oneKg]
            mushroomTestFile = mushroomTestFiles[oneKg]

            mushroomLogPath = mushroomLogPaths[oneKg]
            mushroomLogFile = mushroomLogFiles[oneKg]
            mushroomTCResult = mushroomTCResults[oneKg]

            resultPath = os.path.join(dataPath, oneKg, resultDir)
            if not os.path.exists(resultPath):
                os.makedirs(resultPath)
            for e2vFile in entityEmbeddings:
                mushroom_triple_classification_with_different_pars(threadNum=1,
                                                                   testTripleFile=mushroomTestFile,
                                                                   trainTripeFile=mushroomTrainFile,
                                                                   validTripleFile=mushroomValidFile,
                                                                   mushroomStemRel=defaultStemRel,
                                                                   subspaceDim=defaultSubspaceDim,
                                                                   widthOfPLC=defaultWidthOfPLC,
                                                                   plcFile=mushroomStrucFile,
                                                                   maxTrueTails=defaultMaxTrueTails,
                                                                   enhanceStem=defaultEnhanceStem,
                                                                   e2vecFile=e2vFile, ballMarginLst=ballMarginLst,
                                                                   addDim=defaultAddDim, L0=defaultL0,
                                                                   R0=defaultR0, enrichLevel=defaultEnrichLevel,
                                                                   enlargeMR=defaultEnlargeMR,
                                                                   mPInterSec=defaultMPInterSec, multiParentDicFile="",
                                                                   mushroomGroundTruth=mushroomTestFile,
                                                                   relNballWNFile=relNballWNFile,
                                                                   mushroomPath=mushroomPath,
                                                                   logPath=mushroomLogPath, mainLog=mushroomLogFile,
                                                                   mushroomResultFile=mushroomTCResult)

        if not args.kg == 'all':
            if args.e2v == "TH":
                e2vFiles = [tekeh2vecFiles[args.kg]]
            elif args.e2v == "TE":
                e2vFiles = [tekee2vecFiles[args.kg]]
            elif args.e2v == "TransE":
                e2vFiles = [transe2vecFiles[args.kg]]
            elif args.e2v == "all":
                e2vFiles = [tekeh2vecFiles[args.kg], tekee2vecFiles[args.kg], transe2vecFiles[args.kg]]
            process_one_kg(oneKg=args.kg, entityEmbeddings=e2vFiles)
        else:
            for aKG in KGS:
                if aKG == "all": continue
                if args.e2v == "TH":
                    e2vFiles = [tekeh2vecFiles[aKG]]
                elif args.e2v == "TE":
                    e2vFiles = [tekee2vecFiles[aKG]]
                elif args.e2v == "TransE":
                    e2vFiles = [transe2vecFiles[aKG]]
                elif args.e2v == "all":
                    e2vFiles = [tekeh2vecFiles[aKG], tekee2vecFiles[aKG], transe2vecFiles[aKG]]
                process_one_kg(oneKg=aKG, entityEmbeddings=e2vFiles)


if __name__ == "__main__":
    main()





