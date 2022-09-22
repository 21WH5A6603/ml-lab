#!/usr/bin/env python
# coding: utf-8

# In[5]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
irisData = load_iris()


# In[9]:


x = irisData.data
y = irisData.target
print(x)
print(y)


# In[10]:


X_train, X_test, y_train, y_test = train_test_split( X, y, test_size = 0.2, random_state=42)


# In[11]:



knn = KNeighborsClassifier(n_neighbors=7)


# In[12]:


knn.fit(X_train, y_train)


# In[14]:


knn.predict(X_test)


# In[15]:


knn.score(X_test, y_test)


# In[ ]:




