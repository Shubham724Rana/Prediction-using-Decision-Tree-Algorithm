#!/usr/bin/env python
# coding: utf-8

# # Dataset Information¶
# #### The data set contains 3 classes of 50 instances each, where each class refers to a type of iris plant.
# 
# ### Attribute Information:
# 
# #### sepal length in cm
# #### sepal width in cm
# #### petal length in cm
# #### petal width in cm
# #### class: -- Iris Setosa -- Iris Versicolour -- Iris Virginica

# In[1]:


#importing libraries in python 
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# # Loading the  Dataset

# In[2]:



iris = pd.read_csv("C:/Users/94596/Desktop/sparks/Iris.csv")


# # DataSet Information
# 

# In[3]:


#Basic info about datatype
iris.info()


# In[4]:


iris.head()


# In[5]:


# to display stats about data
iris.describe()


# # Exploratory Data Analysis
# 

# In[6]:


# Histograms
iris["SepalLengthCm"].hist(bins= 50, figsize=(10, 5))


# In[7]:


iris["SepalWidthCm"].hist(bins= 50, figsize=(10, 5))


# In[8]:


iris["PetalLengthCm"].hist(bins= 50, figsize=(10, 5))


# In[9]:


iris["PetalWidthCm"].hist(bins= 50, figsize=(10, 5))


# In[10]:



# scatterplot
colors = ['red', 'orange', 'blue']
species = ['Iris-virginica','Iris-versicolor','Iris-setosa']


# In[11]:


for i in range(3):
    x = iris[iris['Species'] == species[i]]
    plt.scatter(x['SepalLengthCm'], x['SepalWidthCm'], c = colors[i], label=species[i])
plt.xlabel("Sepal Length")
plt.ylabel("Sepal Width")
plt.legend()


# In[12]:


for i in range(3):
    x = iris[iris['Species'] == species[i]]
    plt.scatter(x['PetalLengthCm'], x['PetalWidthCm'], c = colors[i], label=species[i])
plt.xlabel("PetalLengthCm")
plt.ylabel("PetalWidthCm")
plt.legend()


# In[13]:


for i in range(3):
    x = iris[iris['Species'] == species[i]]
    plt.scatter(x['SepalLengthCm'], x['PetalLengthCm'], c = colors[i], label=species[i])
plt.xlabel("SepalLengthCm")
plt.ylabel("PetalLengthCm")
plt.legend()


# In[14]:


for i in range(3):
    x = iris[iris['Species'] == species[i]]
    plt.scatter(x['SepalWidthCm'], x['PetalWidthCm'], c = colors[i], label=species[i])
plt.xlabel("SepalWidthCm")
plt.ylabel("PetalWidthCm")
plt.legend()


# # Coorelation Matrix
# A correlation matrix is a table showing correlation coefficients between variables. Each cell in the table shows the correlation between two variables. The value is in the range of -1 to 1.

# In[15]:


iris.corr()


# In[16]:


correlation_matrix = iris.corr()
figure , axis = plt.subplots(figsize = (7, 7))
sns.heatmap(correlation_matrix, annot = True, ax = axis)


# In[ ]:





# # Data Preprocessing

# In[17]:


X = iris.iloc[:,1:5].values1
y = iris.iloc[:,5].values


# In[18]:



print(y)


# # Label Encoder

# In[19]:



from sklearn.preprocessing import LabelEncoder


# In[20]:


labelEncoder =  LabelEncoder()


# In[21]:


y = labelEncoder.fit_transform(y)


# In[22]:


print(y)


# # Train and Test Split of Data
# 

# In[23]:


from  sklearn.model_selection import train_test_split


# In[24]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)


# # Model Training

# ###  The Decision Tree Classifier

# In[25]:


from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier()
classifier.fit(X_train, y_train)


# In[26]:


print("Accuracy: ", classifier.score(X_test, y_test)* 100)


# # Classifier Prediction

# In[27]:


y_predict = classifier.predict(X_test)
print(y_predict, y_test)


# # Visualization Of Trained Decision Tree Classifier

# In[28]:


#import libraries to plot

from sklearn import tree
from sklearn.tree import plot_tree


# In[29]:


#feature names
X_string = []
for name in iris.iloc[:,1:5].columns.values:
    X_string.append(str(name))
y_string = str(iris.columns.values)

print(X_string, y_string)


# In[30]:


#ploting the Graph
plt.figure(figsize = (15, 7.5))
plot_tree(classifier,  
          filled = True, 
          rounded = True, class_names = y_string,
          feature_names = X_string)


# #  Determining The Accuracy Of Trained Model

# In[31]:


from sklearn.metrics import classification_report, confusion_matrix


# In[32]:


print(classification_report(y_predict, y_test))


# In[ ]:




