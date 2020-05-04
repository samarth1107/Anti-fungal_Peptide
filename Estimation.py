#Header file
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.svm import SVC,NuSVC
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.ensemble import AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn import model_selection
from sklearn.model_selection import GridSearchCV




#importing file from computer as pandas dataframe
Train_rawdata = pd.read_csv('train.csv')
Test_rawdata = pd.read_csv('test.csv') 


#Correction required at 1481th row as label of the row was not a valid number
Train_rawdata.at[1481,'Lable']=1


#Assigning sequence and its label from csv file
X = Train_rawdata['Sequence']
Y = Train_rawdata['Lable'].astype('int')


#To calculate frequency matrix
features_list = {'A':0, 'C':1, 'D':2, 'E':3, 'F':4, 'G':5, 'H':6, 'I':7, 'K':8, 'L':9, 'M':10, 'N':11, 'P':12, 'Q':13, 'R':14, 'S':15, 'T':16, 'V':17, 'W':18, 'Y':19}
to_df=[]
for i in range(X.shape[0]):
    seq = X.iloc[i]
    enc = [0]*20
    for j in seq:
        enc[features_list[j]]+=1
    to_df.append(enc)


#As our training of model will done with frequency matrice
X = pd.DataFrame(to_df, columns = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y'])



#For testing which model perform best in roc_auc 
models = []
models.append(('KNN', KNeighborsClassifier(n_neighbors=5)))
models.append(('NB', GaussianNB()))
models.append(('SVMR', SVC(kernel='rbf',gamma=0.01,C=5)))
models.append(('nuSVMR', NuSVC(kernel='rbf',gamma=0.01)))
models.append(('Random', RandomForestClassifier(n_estimators=130)))
rfc = RandomForestClassifier(n_estimators=100)
models.append(("Adaboost",AdaBoostClassifier(base_estimator=rfc,n_estimators=rfc.n_estimators)))

results=[]
names=[]
for name, model in models:
    kfold = model_selection.KFold(n_splits=5, shuffle=True)
    res = model_selection.cross_val_score(model, X, Y, cv=kfold, scoring='roc_auc')
    results.append(res)
    names.append(name)

fig = plt.figure()                  
fig.suptitle('Models by roc_auc')
ax = fig.add_subplot()
plt.boxplot(results)
ax.set_xticklabels(names)
plt.rcParams['figure.figsize'] = [20,10]
plt.show()

#Result from this graph is stored as "Cross validation score.png" file 
#in that we can see that Adaboost outperform all other models so we will use adaboost with random forest



#to calculate parameters of adaboost 
#at the time of testing 100 was the best n_estimator to get best result in roc graph
param_grid = {'n_estimators': [1,10,30,50,70,100,130,150,170,200,230,250,300]}

ABC = AdaBoostClassifier(base_estimator=RandomForestClassifier(n_estimators=100))

grid = GridSearchCV(ABC, param_grid, scoring='roc_auc')

grid.fit(X,Y)

print(grid.best_score_)
print(grid.best_params_)
