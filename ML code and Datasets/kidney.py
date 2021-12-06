import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
data = pd.read_csv('kidney_disease.csv')
data.classification=data.classification.replace("ckd\t","ckd")
data.drop('id', axis = 1, inplace = True)
data['classification'] = data['classification'].replace(['ckd','notckd'], [1,0])
df = data.dropna(axis = 0)
df.index = range(0,len(df),1)
df['wc']=df['wc'].replace(["\t6200","\t8400"],[6200,8400])
df['pcv']=df['pcv'].astype(int)
df['wc']=df['wc'].astype(int)
df['rc']=df['rc'].astype(float)
object_dtypes = df.select_dtypes(include = 'object')
dictonary = {
        "rbc": {
        "abnormal":1,
        "normal": 0,
    },
        "pc":{
        "abnormal":1,
        "normal": 0,
    },
        "pcc":{
        "present":1,
        "notpresent":0,
    },
        "ba":{
        "notpresent":0,
        "present": 1,
    },
        "htn":{
        "yes":1,
        "no": 0,
    },
        "dm":{
        "yes":1,
        "no":0,
    },
        "cad":{
        "yes":1,
        "no": 0,
    },
        "appet":{
        "good":1,
        "poor": 0,
    },
        "pe":{
        "yes":1,
        "no":0,
    },
        "ane":{
        "yes":1,
        "no":0,
    }
}
df=df.replace(dictonary)
X = df.drop(['classification', 'sg', 'appet', 'rc', 'pcv', 'hemo', 'sod'], axis = 1)
y = df['classification']
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators = 20)
model.fit(X.values, y.values)
import pickle
pickle.dump(model, open('kidney.pkl', 'wb'))

