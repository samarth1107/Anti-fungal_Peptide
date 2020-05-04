#Header files
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier

from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score



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


#Spliting data into training and testing purpose
#We have taken 70% data for training and 30% for testing
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.3, shuffle=True)

base_probs = [0 for i in range(len(Y_test))]

#Fitting data into model
rfc = RandomForestClassifier(n_estimators=300)
model = AdaBoostClassifier(base_estimator=rfc,n_estimators=rfc.n_estimators)
model.fit(X_train,Y_train)

#Here we calculate accuracy and roc of model
model_probs = model.predict_proba(X_test)[:,1]
base_auc = roc_auc_score(Y_test, base_probs)
model_auc = roc_auc_score(Y_test, model_probs)

#roc score
print('Base AUC = {}'.format(base_auc))
print('RF with AdaBoost AUC = {}'.format(model_auc))
#accuracy score
Y_pred = model.predict(X_test)
print("Accuracy of model on test data is : ",accuracy_score(Y_pred, Y_test))


#Plot roc on graph
base_fpr, base_tpr, _ = roc_curve(Y_test, base_probs)
model_fpr, model_tpr, _ = roc_curve(Y_test, model_probs)

plt.plot(base_fpr, base_tpr, linestyle='--', label='Base')
plt.plot(model_fpr, model_tpr, marker='.', label='RF with AdaBoost')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.show()






#saving result from the model in csv file format
#Now we will first calculate frequency matrix of test data
X=Test_rawdata['Sequence']

to_df=[]
for i in range(X.shape[0]):
    seq = X.iloc[i]
    enc = [0]*20
    for j in seq:
        enc[features_list[j]]+=1
    to_df.append(enc)
X = pd.DataFrame(to_df, columns = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y'])

#Saving data according to sample.csv format
Pred_result=(model.predict(X))
IDs=Test_rawdata['ID'].values.tolist()
import csv
with open('result.csv','w',newline='') as f:
  thewriter=csv.writer(f)
  thewriter.writerow(['ID','Label'])
  for i in range(0,len(Pred_result)):
    thewriter.writerow([IDs[i],Pred_result[i]])
