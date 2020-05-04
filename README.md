# Anti fungal peptide prediction

-----> File description 

train.csv                   - Train data from kaggle  

test.csv                    - Test data from kaggle for we have to calculate model  

result.csv                  - Result from mainCode.py (from our model)  

Estimation.py               - In this python code we try to select best possible model with respect   
                                to roc and then estimate its best parameter    
                                
mainCode.py                 - In this commented python we train, test, evaluate its accuracy and get   
                                result from the model 

Cross_Validation_score.png  - Cross validation score from model selection in estimator.py 
Roc_graph.png               - Roc graph 



-----> Main idea
In machine learning model we cannot give input in character but in numbers only and input size of
these number should also be constant,here we have to train model with protein sequence which is 
in character so we can convert these numbers into relatable numbers ie which hold the importance of 
characters size of the sequence to do this we have following model :-
1. Frequency Matrice
2. Binary Array Conversion
3. Composition matrice using pfeatures

From above option Frequency Matrices give us the best result in this we have array of 20 length and 
each index store the Frequency of that corresponding character like 0th index represent A character 
so number at 0th index will represent frequency of A in the entire sequence.
As it calculate the frequency it store the relevance to sequence
As the frequency matice of all sequence is of length 20 so the input length of sequence is constant.

-----> Fitting frequency matrices into model  
after calculating the frequency matrice we run it on model selection function which plot graph of 
model vs roc_auc score which is store in Cross_Validation_score.png, from this graph we can clearly 
see that the adaboost with random forest perform best then other classifier. There are many classifier
available but we choose those type of classifier which are best for classifying sequence.

After choosing adaboost with random forest as our model now we have to calculate its best parameter
for this we used grid search function with roc as parameter which gives us that best
n_estimator is 300 from list of [1,10,30,50,70,100,130,150,170,200,230,250,300] estimators  

Code for this 2 is stored in the Estimation.py File 


After getting model and its best parameter we fit frequency model in model and calculate our result
for test data from kaggle in result.csv file on kaggle  
accuracy score on internal data (test data by spliting train data from kaggle) is 89% 
AUC score for roc on internal data is approx 93%  
accuracy score on external data (test file from kaggle) 92% approx  
