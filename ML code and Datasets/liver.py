import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
data = pd.read_csv('liver.csv')
data['Albumin_and_Globulin_Ratio'] = data['Albumin_and_Globulin_Ratio'].fillna(0.947064)
data['Dataset'] = data['Dataset'].replace([2,1],[1,0])
data = pd.get_dummies(data, columns = ['Gender'], drop_first = True)
X = data.drop('Dataset', axis = 1)
y = data['Dataset']
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=20)
model.fit(X.values, y.values)
import pickle
pickle.dump(model, open('liver.pkl', 'wb'))

