#!/usr/bin/env python
# coding: utf-8

# # Task 3 : Prediction using Decision Tree Algorithm
# Decision Trees are versatile Machine Learning algorithms that can perform both classification and regression tasks, and even multioutput tasks.For the given ‘Iris’ dataset, I created the Decision Tree classifier and visualized it graphically. The purpose of this task is if we feed any new data to this classifier, it would be able to predict the right class accordingly.  

# In[4]:


get_ipython().system('pip install pydot')


# In[5]:


# Importing the required Libraries

from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.model_selection import train_test_split
import sklearn.metrics as sm

import pandas as pd
import numpy as np
import seaborn as sns

import matplotlib.pyplot as plt
import pydot
from IPython.display import Image


# 1 - Loading the Dataset

# In[6]:


# Loading Dataset
iris = load_iris()
X=iris.data[:,:] 
y=iris.target


# 2 - Exploratory Data Analysis

# In[7]:


#Input data 

data=pd.DataFrame(iris['data'],columns=["Petal length","Petal Width","Sepal Length","Sepal Width"])
data['Species']=iris['target']
data['Species']=data['Species'].apply(lambda x: iris['target_names'][x])

data.head()


# In[8]:


data.shape


# In[9]:


data.describe()


#  3 - Data Visualization comparing various features

# In[10]:


# Input data Visualization
sns.pairplot(data)


# In[11]:


# Scatter plot of data based on Sepal Length and Width features
sns.FacetGrid(data,hue='Species').map(plt.scatter,'Sepal Length','Sepal Width').add_legend()
plt.show()

# Scatter plot of data based on Petal Length and Width features
sns.FacetGrid(data,hue='Species').map(plt.scatter,'Petal length','Petal Width').add_legend()
plt.show()


# 4 - Decision Tree Model Training

# In[13]:


# Model Training
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=1) 
tree_classifier = DecisionTreeClassifier()
tree_classifier.fit(X_train,y_train)
print("Training Complete.")
y_pred = tree_classifier.predict(X_test)


# 5 - Comparing the actual and predicted flower classification

# In[14]:


df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred}) 
df 


#  6 - Visualizing the Trained Model

# In[16]:


#Visualizing the trained Decision Tree Classifier taking all 4 features in consideration

export_graphviz(
        tree_classifier,
        out_file="img\desision_tree.dot",
        feature_names=iris.feature_names[:],
        class_names=iris.target_names,
        rounded=True,
        filled=True
)

(graph,) = pydot.graph_from_dot_file('img\desision_tree.dot')
graph.write_png('img\desision_tree.png')

Image(filename='img\desision_tree.png') 


# In[ ]:




