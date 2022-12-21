#!/usr/bin/env python
# coding: utf-8

# # Task 1 : Prediction using Supervised Machine Learning
# 
# In this regression task I tried to predict the percentage of marks that a student is expected to score based upon the number of hours they studied.
# 
# This is a simple linear regression task as it involves just two variables.  
# 

# In[16]:


# Importing the required libraries
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np  


# 1- Reading data from source 

# In[17]:


# Reading data from remote link
url = r"https://raw.githubusercontent.com/AdiPersonalWorks/Random/master/student_scores%20-%20student_scores.csv"
s_data = pd.read_csv(url)
print("Data import successful")

s_data.head(10)


# 2- data visualization

# In[18]:


# Plotting the distribution of scores
s_data.plot(x='Hours', y='Scores', style='o')  
plt.title('Hours vs Percentage')  
plt.xlabel('Hours Studied')  
plt.ylabel('Percentage Score')  
plt.show()


# we can see a positive linear relation between the two variables 

# 3- Data Preprocessing

# In[19]:


X = s_data.iloc[:, :-1].values  
y = s_data.iloc[:, 1].values 


# 3- Model training

# In[20]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0) 
regressor = LinearRegression()  
regressor.fit(X_train.reshape(-1,1), y_train) 

print("Training complete.")


#  5 - Plotting the Line of regression

# In[21]:


# Plotting the regression line
line = regressor.coef_*X+regressor.intercept_

# Plotting for the test data
plt.scatter(X, y)
plt.plot(X, line,color='red');
plt.show()


# 6 - Making Predictions

# In[22]:


# Testing data
print(X_test)
# Model Prediction 
y_pred = regressor.predict(X_test)


# 7 - Comparing Actual result to the Predicted Model result

# In[23]:


# Comparing Actual vs Predicted
df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred}) 
df 


# In[24]:


#Estimating training and test score
print("Training Score:",regressor.score(X_train,y_train))
print("Test Score:",regressor.score(X_test,y_test))


# In[25]:


# Plotting the Bar graph to depict the difference between the actual and predicted value

df.plot(kind='bar',figsize=(5,5))
plt.grid(which='major', linewidth='0.5', color='red')
plt.grid(which='minor', linewidth='0.5', color='blue')
plt.show()


# In[26]:


# Testing the model with our own data
hours = 9.25
test = np.array([hours])
test = test.reshape(-1, 1)
own_pred = regressor.predict(test)
print("Nbr of Hours = {}".format(hours))
print("Predicted Score = {}".format(own_pred[0]))


# 8 - Evaluating the model

# In[27]:


from sklearn import metrics  
print('Mean Absolute Error:',metrics.mean_absolute_error(y_test, y_pred)) 
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
print('R-2:', metrics.r2_score(y_test, y_pred))


# R-2 gives the score of model fit and in this case we have R-2 = 0.9454906892105355 which is actually a great score for this model.

# # Conclusion
# I was successfully able to carry-out Prediction using Supervised ML task and was able to evaluate the model's performance on various parameters.

# In[ ]:




