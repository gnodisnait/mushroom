import os
import decimal

################################
#### put your own data path ####
################################
dataPath = "/Users/tdong/data/mushroom"

########################################################################
### name the directory for putting results of triple classification  ###
########################################################################
resultDir = "tripleClassification"

#
KGS = ["Freebase13", "Wordnet11", "Wordnet18", "all"]
e2vFlags = ['TransE', 'TE', 'TH', 'all']
NBallLinks = {
    "Freebase13": "https://ndownloader.figshare.com/files/13548377",
    "Wordnet11": "https://ndownloader.figshare.com/files/13548311",
    "Wordnet18": "https://ndownloader.figshare.com/files/13548311"
}

Precision = 500

e2vDic = {
        'TH': 'TEKE_H',
        'TE': 'TEKE_E',
        'TransE': 'TransE'
        }

relNballWNFile = os.path.join(dataPath, "relationNBall.txt")
word2vecFile = os.path.join(dataPath, "glove.6B.50d.txt")

tekee2vecFiles = {"Freebase13": os.path.join(dataPath, "Freebase13/entity2vec.TEKEE.50"),
                  "Wordnet11": os.path.join(dataPath, "Wordnet11/entity2vec.TEKEE.50"),
                  "Wordnet18": os.path.join(dataPath, "Wordnet18/entity2vec.TEKEE.50"),
                  }

tekeh2vecFiles = {"Freebase13": os.path.join(dataPath, "Freebase13/entity2vec.TEKEH.100"),
                  "Wordnet11": os.path.join(dataPath, "Wordnet11/entity2vec.TEKEH.50"),
                  "Wordnet18": os.path.join(dataPath, "Wordnet18/entity2vec.TEKEH.50"),
                  }

transe2vecFiles ={"Freebase13": os.path.join(dataPath, "Freebase13/entity2vec.TransE.50"),
                  "Wordnet11": os.path.join(dataPath, "Wordnet11/entity2vec.TransE.50"),
                  "Wordnet18": os.path.join(dataPath, "Wordnet18/entity2vec.TransE.50"),
                  }

mushroomStrucFiles = {"Freebase13": os.path.join(dataPath, "Freebase13/mushroom_structure.txt"),
                      "Wordnet11": os.path.join(dataPath, "Wordnet11/mushroom_structure.txt"),
                      "Wordnet18": os.path.join(dataPath, "Wordnet18/mushroom_structure.txt"),
                      }

mushroomTrainFiles = {
    "Freebase13": os.path.join(dataPath, "Freebase13/train_decoded_mushroom.txt.clean"),
    "Wordnet11": os.path.join(dataPath, "Wordnet11/train_decoded_mushroom.txt.clean"),
    "Wordnet18": os.path.join(dataPath, "Wordnet18/train_decoded_mushroom.txt.clean")
}

mushroomValidFiles = {
    "Freebase13": os.path.join(dataPath, "Freebase13/valid_decoded_mushroom.txt.clean"),
    "Wordnet11": os.path.join(dataPath, "Wordnet11/valid_decoded_mushroom.txt.clean"),
    "Wordnet18": os.path.join(dataPath, "Wordnet18/valid_decoded_mushroom.txt.clean")
}

mushroomTestFiles = {
    "Freebase13": os.path.join(dataPath, "Freebase13/test_decoded_mushroom.txt.clean"),
    "Wordnet11": os.path.join(dataPath, "Wordnet11/test_decoded_mushroom.txt.clean"),
    "Wordnet18": os.path.join(dataPath, "Wordnet18/test_decoded_mushroom.txt.clean")
}

mushroomLogPaths = {
    "Freebase13": os.path.join(dataPath, "Freebase13", resultDir, "MRlog"),
    "Wordnet11": os.path.join(dataPath, "Wordnet11", resultDir, "MRlog"),
    "Wordnet18": os.path.join(dataPath, "Wordnet18", resultDir, "MRlog")
}

mushroomLogFiles = {
    "Freebase13": os.path.join(dataPath, "Freebase13", resultDir, "TClog.txt"),
    "Wordnet11": os.path.join(dataPath, "Wordnet11", resultDir, "TClog.txt"),
    "Wordnet18": os.path.join(dataPath, "Wordnet18", resultDir, "TClog.txt")
}

mushroomTCResults = {
    "Freebase13": os.path.join(dataPath, "Freebase13", resultDir, "MRTCresult"),
    "Wordnet11": os.path.join(dataPath, "Wordnet11", resultDir, "MRTCresult"),
    "Wordnet18": os.path.join(dataPath, "Wordnet18", resultDir, "MRTCresult")
}

mushroomPaths = {
    "Freebase13": os.path.join(dataPath, "Freebase13", resultDir, "mushrooms"),
    "Wordnet11": os.path.join(dataPath, "Wordnet11", resultDir, "mushrooms"),
    "Wordnet18": os.path.join(dataPath, "Wordnet18", resultDir, "mushrooms")
}


defaultGammaLst = [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7,1.8, 1.9, 2.0, 2.1, 2.2, 2.3]
defaultStemLst = [0, 1 , 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16 , 17]
ballMarginLst = [1.0, 0.9, 0.8, 0.7, 0.6, 1.1, 1.2, 1.3, 1.4, 1.5,  1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3] # donot change the order

defaultStemRel = "tr_contain1"
defaultSubspaceDim = 20
defaultWidthOfPLC = 30
defaultMaxTrueTails = 60
defaultEnhanceStem = False
defaultAddDim = [512] * 100
defaultL0 = decimal.Decimal(1e+100)
defaultR0 = decimal.Decimal(1e-200)
defaultEnrichLevel = -1
defaultEnlargeMR = -1
defaultMPInterSec = False
