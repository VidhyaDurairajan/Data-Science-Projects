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

# ANALYSIS 1: DECISION TREES

#Splitting data for training and testing

import numpy
from sklearn.model_selection import train_test_split

numpy.random.seed(1234)

(training_inputs,testing_inputs,training_classes,testing_classes) = train_test_split(scaled_features,
                                                                                     label, 
                                                                                     train_size=0.75,
                                                                                     random_state=1)

#Initializing a Decision Tree Classifier
                                                                                        
from sklearn.tree import DecisionTreeClassifier

clf= DecisionTreeClassifier(random_state=1)

# Train the classifier on the training set

clf.fit(training_inputs, training_classes)

#Displaying the resulting Decision Tree

from IPython.display import Image  
from sklearn.externals.six import StringIO  
from sklearn import tree
from pydotplus import graph_from_dot_data 
import os     

os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'

dot_data = StringIO()  
tree.export_graphviz(clf, out_file=dot_data,  
                         feature_names=feature_names)  
graph = graph_from_dot_data(dot_data.getvalue())  
Image(graph.create_png()) 

# Accuracy obtained with Decision tree

clf.score(testing_inputs, testing_classes)

# With K - fold Cross Validation

from sklearn.model_selection import cross_val_score

clf = DecisionTreeClassifier(random_state=1)

cv_scores = cross_val_score(clf, scaled_features, label, cv=10)

cv_scores.mean()

# With Random Forest

from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(n_estimators=10, random_state=1)
cv_scores = cross_val_score(clf, scaled_features, label, cv=10)

cv_scores.mean()