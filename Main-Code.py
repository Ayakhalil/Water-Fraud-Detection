#!/usr/bin/env python
# coding: utf-8

# In[1]:


import mysql.connector
import pandas as pd
from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.svm import SVC
from sklearn.svm import OneClassSVM
from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as mtp  
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import numpy as np
import math


# In[2]:


mydb = mysql.connector.connect(
  host="localhost",
  user="root",
  passwd="1234",
  database="gproject"
)

mycursor = mydb.cursor()


# In[3]:


df = pd.read_csv (r'C:\Users\jack_\OneDrive\Desktop\final data.csv')

data = df.drop('seq' , axis=1)


# In[4]:


y = data.flag
x = data.drop('flag' , axis=1)


# In[6]:


data['flag'].value_counts()


# In[7]:


sb.countplot(x='flag' , data=data , palette = 'hls')
plt.show()
plt.savefig('count_plot')


# In[8]:


x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.10, random_state = 10)


# In[39]:


modellogreg= LogisticRegression(solver='lbfgs', max_iter=9999)
modellogreg.fit(x_train, y_train)


# In[59]:


modelrandomforests = RandomForestClassifier(n_estimators=100)
modelrandomforests.fit(x_train, y_train)


# In[60]:


modelknn = KNeighborsClassifier()
modelknn.fit(x_train, y_train)


# In[61]:


modelsvm=SVC(kernel='linear')
modelsvm.fit(x_train, y_train)


# In[62]:


logreg = modellogreg.predict(x_test)
logreg


# In[63]:


randomforests = modelrandomforests.predict(x_test)
randomforests


# In[64]:


knn= modelknn.predict(x_test)
knn


# In[65]:


svm= modelsvm.predict(x_test)
svm


# In[114]:


sql = "select seq from dataformysql where consumption_1 = %s AND consumption_2 = %s AND consumption_3 = %s AND consumption_4 = %s AND consumption_5 = %s"
resultlogreg = []
for res in range(len(logreg)):
    if logreg[res] == 1:
        con1 = x_test.iloc[res].consumption_1
        con2 = x_test.iloc[res].consumption_2
        con3 = x_test.iloc[res].consumption_3
        con4 = x_test.iloc[res].consumption_4
        con5 = x_test.iloc[res].consumption_5  
        
        cons = (int(con1), int(con2), int(con3), int(con4), int(con5) )     
        mycursor.execute(sql, cons)
        myresult = mycursor.fetchone()
        for i in myresult:
            resultlogreg.append(i)
            
resultlogreg          
            


# In[116]:


sql = "select seq from dataformysql where consumption_1 = %s AND consumption_2 = %s AND consumption_3 = %s AND consumption_4 = %s AND consumption_5 = %s"
resultrandomforests = []
for res in range(len(randomforests)):
    if randomforests[res] == 1:
        con1 = x_test.iloc[res].consumption_1
        con2 = x_test.iloc[res].consumption_2
        con3 = x_test.iloc[res].consumption_3
        con4 = x_test.iloc[res].consumption_4
        con5 = x_test.iloc[res].consumption_5  
        
        cons = (int(con1), int(con2), int(con3), int(con4), int(con5) )     
        mycursor.execute(sql, cons)
        myresult = mycursor.fetchone()
        for i in myresult:
            resultrandomforests.append(i)
            
print(resultrandomforests)            


# In[117]:


sql = "select seq from dataformysql where consumption_1 = %s AND consumption_2 = %s AND consumption_3 = %s AND consumption_4 = %s AND consumption_5 = %s"
resultknn = []
for res in range(len(knn)):
    if knn[res] == 1:
        con1 = x_test.iloc[res].consumption_1
        con2 = x_test.iloc[res].consumption_2
        con3 = x_test.iloc[res].consumption_3
        con4 = x_test.iloc[res].consumption_4
        con5 = x_test.iloc[res].consumption_5  
        
        cons = (int(con1), int(con2), int(con3), int(con4), int(con5) )     
        mycursor.execute(sql, cons)
        myresult = mycursor.fetchone()
        for i in myresult:
            resultknn.append(i)
            
print(resultknn)   


# In[120]:


sql = "select seq from dataformysql where consumption_1 = %s AND consumption_2 = %s AND consumption_3 = %s AND consumption_4 = %s AND consumption_5 = %s"
resultsvm = []
for res in range(len(svm)):
    if svm[res] == 1:
        con1 = x_test.iloc[res].consumption_1
        con2 = x_test.iloc[res].consumption_2
        con3 = x_test.iloc[res].consumption_3
        con4 = x_test.iloc[res].consumption_4
        con5 = x_test.iloc[res].consumption_5  
        
        cons = (int(con1), int(con2), int(con3), int(con4), int(con5) )     
        mycursor.execute(sql, cons)
        myresult = mycursor.fetchone()
        for i in myresult:
            resultsvm.append(i)
            
print(resultsvm)   


# In[115]:


plt.scatter(data.consumption_1, y )
plt.show()


# In[109]:


acclogreg = modellogreg.score(x_test,y_test)
print("accuracy = ",acclogreg * 100)
cmlogreg = confusion_matrix(y_test, logreg)
print("confusion matrix :\n",cmlogreg)
print(classification_report(y_test,logreg))


# In[110]:


accrandomforests = modelrandomforests.score(x_test,y_test)
print("accuracy = ",accrandomforests * 100)
cmrandomforests = confusion_matrix(y_test, randomforests)
print("confusion matrix :\n",cmrandomforests)
print(classification_report(y_test,randomforests))


# In[111]:


accsvm = modelsvm.score(x_test,y_test)
print("accuracy = ",accsvm * 100)
cmsvm = confusion_matrix(y_test, svm)
print("confusion matrix :\n", cmsvm)
print(classification_report(y_test,svm))


# In[112]:


accknn = modelknn.score(x_test,y_test)
print("accuracy = ", accknn * 100)
cmknn = confusion_matrix(y_test, knn)
print("confusion matrix :\n",cmknn)
print(classification_report(y_test,knn))


# In[ ]:




