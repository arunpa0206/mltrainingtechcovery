from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials

#load dataset
iris = datasets.load_iris()
X = iris.data
y = iris.target

#for a given set of parameters check accuracy
def hyperopt_train_test(params):
    clf = KNeighborsClassifier(**params)
    #cross_val_score estimates the expected accuracy of your
    # model on out-of-training data
    return cross_val_score(clf, X, y).mean()

#define the search space for the hyper parametr K
space4knn = {
    'n_neighbors': hp.choice('n_neighbors', range(1,100))
}
#this function returns the loss function which is negative Accuracy
#minimize negative accuracy by choosing right hyperparametrs
#this will maximise the accuracy
def f(params):
    acc = hyperopt_train_test(params)
    return {'loss': -acc, 'status': STATUS_OK}
trials = Trials()
#optimization algoriyhm used is tpe (tree parzen esimator)
best = fmin(f, space4knn, algo=tpe.suggest, max_evals=100, trials=trials)
print ('best:')
print (best)
