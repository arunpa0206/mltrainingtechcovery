import pandas as pd
import numpy as np

from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
np.random.seed(2)

data = pd.read_csv('./creditcard.csv')
print('sample records:')
print(data.head())

from sklearn.preprocessing import StandardScaler
data['normalizedAmount'] = StandardScaler().fit_transform(data['Amount'].values.reshape(-1,1))
data = data.drop(['Amount'],axis=1)
data = data.drop(['Time'],axis=1)

X = data.iloc[:, data.columns != 'Class']
y = data.iloc[:, data.columns == 'Class']
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.3, random_state=0)

print('X_train shape:',X_train.shape)
print('X_test shape:', X_test.shape)


from sklearn.ensemble import RandomForestClassifier
random_forest = RandomForestClassifier(n_estimators=5)


random_forest.fit(X_train,y_train.values.ravel())
y_pred = random_forest.predict(X_test)
random_forest.score(X_test,y_test)
cnf_matrix = confusion_matrix(y_test,y_pred)
labels = [0,1]
sns.heatmap(cnf_matrix, annot=True, cmap="YlGnBu", fmt=".3f", xticklabels=labels, yticklabels=labels)
plt.show()

y_pred = random_forest.predict(X)
cnf_matrix = confusion_matrix(y,y_pred.round())

sns.heatmap(cnf_matrix, annot=True, cmap="YlGnBu", fmt=".3f", xticklabels=labels, yticklabels=labels)
plt.show()
