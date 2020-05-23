from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
#import numpy as np
from statistics import mode

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


cancer = load_breast_cancer()
x_train, x_test, y_train, y_test = train_test_split(cancer.data, cancer.target, stratify=cancer.target, random_state=42)


model1 = LogisticRegression(random_state=1, solver='lbfgs',max_iter=7600)


model1.fit(x_train,y_train)
print(' LogisticRegression accuracy:',model1.score(x_test,y_test))
