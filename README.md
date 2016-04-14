# UCLirdm2016

## Collaborative Filtering via low rank matrix factorisation

Code for UCL IRDM group project, 2016

Rikk Hill
Alberto Martin Izquierdo
Kan Yin Yee

### Initial setup
Requires the [MovieLens 1M dataset and genome dataset](http://grouplens.org/datasets/movielens/) downloaded and unzipped into a directory named `data`

### Manual
**bicv.py**

Runs grid search over a range of values for parameter K.

**factorise.py**

Takes the 1M dataset, builds it into a matrix and carries out NMF. Takes an optional command line parameter for K, the number of bases of the factorisation; defaults to 10

**semantics.py**

Takes results from `factorise.py` and finds high-relevance tags corresponding to each basis. Takes an optional command line parameter for K, the number of bases; defaults to 10

**recommend.py**

Selects a random user, displays that user's highest factorisation bases, and carries out k-NN search against the movie basis vectors to recommend films for that user. Takes optional command line parameters for K, the number of bases, and N, the number of movies to recommend; defaults to 10 and 5 respectively



