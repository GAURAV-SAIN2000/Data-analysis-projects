#!/usr/bin/env python
# coding: utf-8

# In[1]:


#importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


#importing the data
dataset = pd.read_csv('Ecommerce Customers')


# In[3]:


#printing the top 5 rows of our dataset using the head function
dataset.head()


# In[4]:


dataset.shape


# In[5]:


dataset.describe()


# In[6]:


dataset.info()


# ### -------Bivariate Analysis----

# In[7]:


## finding out the extent and nature of relationship between two variables. Ideally we are checking the relationship of the
## independent variables on the dependent variables


# In[8]:


sns.jointplot(x='Time on Website',y='Yearly Amount Spent',data=dataset)


# In[9]:


sns.jointplot(x='Time on App',y='Yearly Amount Spent',data=dataset)


# In[10]:


sns.jointplot(x='Avg. Session Length',y='Yearly Amount Spent',data=dataset)


# In[11]:


sns.jointplot(x='Length of Membership',y='Yearly Amount Spent',data=dataset)


# ### Length of membership has the maximum impact on the dependent variable

# In[12]:


#traing and testing feature sets
X = dataset[['Avg. Session Length','Length of Membership','Time on App','Time on Website']]
y = dataset['Yearly Amount Spent']


# In[13]:


#train-test split
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25,random_state=42)


# In[14]:


from sklearn.linear_model import LinearRegression


# In[15]:


lm = LinearRegression()


# In[16]:


#fitting the training data to the model
lm.fit(X_train,y_train)


# In[17]:


lm.coef_       #so,length of membership has the maximum impact on the DV,as seen with the maximum coefficient value


# In[18]:


#generate the predictions
predictions = lm.predict(X_test)


# In[19]:


#calculating Rsquare
from sklearn.metrics import r2_score
r2= r2_score(y_test,predictions)
print(r2)


# ### Using Ridge,Lasso,ElasticNet models

# #### Normalization is important for regularization. this is because the scale of the variables affect how much regularization needs to be applied to a specific variable. If one variable is in a very large scale,then regularization will have very little impact on that variable.

# In[20]:


import sklearn


# In[21]:


from sklearn.linear_model import Ridge
ridgereg= Ridge(alpha=0.001,normalize=True)
ridgereg.fit(X_train,y_train)


# In[25]:


print('R2 value:{}'.format(ridgereg.score(X_test,y_test)))

#the model.score method ensures we do not need to provide the predictions made by the model
#externally,the predictions are calculated internally using X_test


# In[23]:


#Lasso
from sklearn.linear_model import Lasso
Lassoreg= Lasso(alpha=0.001,normalize=True)
Lassoreg.fit(X_train,y_train)


# In[27]:


print('R2 value:{}'.format(Lassoreg.score(X_test,y_test)))


# In[28]:


#Elastic Net 
from sklearn.linear_model import ElasticNet
Elastic= ElasticNet(alpha=0.001,normalize=True)
Elastic.fit(X_train,y_train)


# In[29]:


print('R2 value:{}'.format(Elastic.score(X_test,y_test)))


# ### If Alpha is close to zero,the ridge term itself is very small and thus the final error is based on RSS alone. If alpha is too large,the impact of Shrinkage grows and the coefficients beta values tends to zero

# In[ ]:




