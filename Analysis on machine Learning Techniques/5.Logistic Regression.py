#Preparing Data

from IPython import get_ipython
ipy = get_ipython()
if ipy is not None:
    ipy.run_line_magic('matplotlib', 'inline')

import numpy as np
import pandas as pd

df = pd.read_csv("E:/NJIT/Data Science/DataScience/DataScience-Python3/projects/Analysis on machine Learning Techniques/mammographic_masses.data.txt", na_values=['?'],names = ['BI_RADS','age','shape', 'margin', 'density','severity'])
df.head()

#Understanding the Data

df.describe()

# Extracting and Eliminating the data with NaN

df.loc[(df['age'].isnull())|
       (df['shape'].isnull())|
       (df['margin'].isnull())|
       (df['density'].isnull())]
       
df.dropna(inplace=True)
df.describe()

#Splitting data into Features and label

feature = df[['age','shape','margin','density']].values
feature_names=['age','shape','margin','density']
label = df['severity'].values
label

#Preprocessing : Normalizing data with standard Scaler

from sklearn import preprocessing

scaler = preprocessing.StandardScaler()
scaled_features = scaler.fit_transform(feature)
scaled_features

#  ANALYSIS 4: Logistic Regression

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

clf = LogisticRegression()
cv_scores = cross_val_score(clf, scaled_features,label, cv=10)
cv_scores.mean()
