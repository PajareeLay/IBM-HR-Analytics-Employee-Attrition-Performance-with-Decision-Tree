#!/usr/bin/env python
# coding: utf-8

# In[6]:


import matplotlib.pyplot as plt
import pydotplus
import pandas as pd
from sklearn.tree import DecisionTreeClassifier 
import graphviz
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

#prepare database
uri = r'C:\Users\PAJAREE\Downloads\WA_Fn-UseC_-HR-Employee-Attrition.csv'
df = pd.read_csv(uri)
col = ['EmployeeCount','EmployeeNumber','Over18','StandardHours']
a = df.drop(col,axis = 1 ,inplace= True)

X = df.drop('Attrition',axis=1)
y=  df.Attrition

#LabelEncode
def labelEncode(data,column):
    for i in column:
        lb = LabelEncoder().fit_transform(data[i])
        data[i] = lb
        
column = ['Department','BusinessTravel','EducationField','Department','Gender','JobRole','MaritalStatus','OverTime' ]
labelEncode(df,column)

y_le = LabelEncoder()
y = y_le.fit_transform(df.Attrition)
df['Attrition'] = y
df

#split 70-30 ,random state = 42 
X = df.drop('Attrition',axis=1)
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state = 42)

#check total X ,X_train, X_test
print('total X:  {}'.format(len(X)))
print('total X_Train:  {}'.format(len(X_train)))
print('total X_Test:  {}'.format(len(X_test)))

#model
model = DecisionTreeClassifier()
model.fit(X_train,y_train)

#diagram

from sklearn.tree import export_graphviz
from IPython.display import Image


class_names = list(y_le.classes_)
dot_data =  export_graphviz(model,out_file = None , class_names = class_names,filled =True, rounded = True)
graph = pydotplus.graph_from_dot_data (dot_data)
Image(graph.create_png())


# In[4]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.metrics import roc_auc_score,roc_curve,accuracy_score,precision_score,confusion_matrix,recall_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score,cross_val_predict
from sklearn.model_selection import train_test_split

#train Classification 
y_pre = model.predict(X_train)
y_pre

import scikitplot as skplot
skplot.metrics.plot_confusion_matrix(y_train,y_pre,normalize = False)
plt.show()
print('Accuracy_score:{} %'.format(accuracy_score(y_train,y_pre)*100))
print(classification_report(y_train,y_pre))


# In[5]:


#test Classification 
y_predict = model.predict(X_test)
import scikitplot as skplot
skplot.metrics.plot_confusion_matrix(y_test,y_predict,normalize = False)
plt.show()
print('Accuracy_score:{} %'.format(accuracy_score(y_test,y_predict)*100))

print(classification_report(y_test,y_predict))

