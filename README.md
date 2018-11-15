# Triple Classification Using Regions and Fine-Grained Entity Typing

## Install system

```
$ git clone https://github.com/gnodisnait/mushroom.git
$ cd mushroom
$ virtualenv venv
$ source venv/bin/activate
(venv) $ pip install -r requirements.txt
```

## Download datasets

open `mushroom/config.py` file, set `dataPath` to the absolute path of your datasets for this project;
download datasets
```
(venv) $ python mushroom.py --func download
```

## Triple classification

```
(venv) $ python mushroom.py --func tc --kg Wordnet18|Wordnet11|Freebase13|all --e2v TH|TE|TransE|all
```

## View results

#### visualize contribution of gamma to precision|recall|accuracy
```
(venv) $ python mushroom.py --vis_gamma precision|recall|accuracy  --kg Wordnet18|Wordnet11|Freebase13
```

#### visualize contribution of length-of-type-chain to precision|recall|accuracy
```
(venv) $ python mushroom.py --vis_length precision|recall|accuracy  --kg Wordnet18|Wordnet11|Freebase13
```

#### visualize max precision|recall|accuracy is reached by what length-of-type-chain with which-gamma
```
(venv) $ python mushroom.py --vis_mg precision|recall|accuracy --kg Wordnet18|Wordnet11|Freebase13  --legendLoc lower right
```
