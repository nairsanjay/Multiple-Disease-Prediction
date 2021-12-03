
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
data = pd.read_csv('cancer.csv')
dataset = data
dataset['diagnosis'].replace(['M','B'], [1,0], inplace = True)
dataset.drop('Unnamed: 32',axis = 1, inplace = True)
dataset.drop(['id','symmetry_se','smoothness_se','texture_se','fractal_dimension_mean'], axis = 1, inplace = True)
X = dataset.drop('diagnosis', axis = 1)
y = dataset['diagnosis']
from sklearn.ensemble import RandomForestClassifier
classifier=RandomForestClassifier(n_jobs=-1, n_estimators=200,bootstrap= True,criterion='gini',max_depth=20,max_features=8,min_samples_leaf= 1)
classifier.fit(X.values, y.values)

import pickle
pickle.dump(classifier, open('cancer.pkl', 'wb'))

