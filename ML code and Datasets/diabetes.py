import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
data = pd.read_csv('data.csv')
X = data.iloc[:,:-1]
y = data['Outcome']
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=20)
model.fit(X.values, y.values)
import pickle
pickle.dump(model, open("diabetes.pkl",'wb'))

