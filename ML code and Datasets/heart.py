heartimport numpy as np
import pandas as pd
import matplotlib.pyplot as plt
data = pd.read_csv('heart.csv')
dataset = data.copy()
X = dataset.drop(['target'], axis = 1)
y = dataset['target']
from sklearn.ensemble import RandomForestClassifier
classifier=RandomForestClassifier(n_jobs=-1, n_estimators=400,bootstrap= False,criterion='gini',max_depth=5,max_features=3,min_samples_leaf= 7)
classifier.fit(X.values, y.values)
import pickle
pickle.dump(classifier, open('heart.pkl', 'wb'))