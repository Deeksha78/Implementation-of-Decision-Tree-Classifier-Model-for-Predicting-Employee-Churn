# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required libraries.
2. Upload and read the dataset.
3. Check for any null values using the isnull() function.
4. From sklearn.tree import DecisionTreeClassifier and use criterion as entropy.
5. Find the accuracy of the model and predict the required values by importing the required module from sklearn.

## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: Deeksha P
RegisterNumber:  212222040031
*/
```
```
import pandas as pd
data=pd.read_csv("/content/Employee.csv")

data.head()

data.info()

data.isnull().sum()

data["left"].value_counts()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()

data["salary"]=le.fit_transform(data["salary"])
data.head()

x=data[["satisfaction_level","last_evaluation","number_project","average_montly_hours","time_spend_company","Work_accident","promotion_last_5years","salary"]]
x.head()

y=data["left"]

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=100)

from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(criterion="entropy")
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)

from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy

dt.predict([[0.5,0.8,9,260,6,0,1,2]])
```
## Output:
Initial data set

![image](https://github.com/Deeksha78/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/128116204/1773bfd7-e1b4-43d0-b660-4d1463ebc483)

Data Info

![image](https://github.com/Deeksha78/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/128116204/9cea1613-cf7a-45db-8ab8-27c835ee13a7)

Optimization of null values

![image](https://github.com/Deeksha78/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/128116204/26ba46fe-9e08-48f5-91ea-601049829e3d)

Assignment of X and Y values

![image](https://github.com/Deeksha78/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/128116204/e9f973fd-03cd-41e7-b849-ed3a9e5d8f7f)

![image](https://github.com/Deeksha78/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/128116204/da4568e5-0f1c-447c-91f2-5673e3d776fe)

Converting string literals to numerical values using label encoder

![image](https://github.com/Deeksha78/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/128116204/137fa543-4d5f-4bc6-88aa-9bc49a9943c6)

Accuracy

![image](https://github.com/Deeksha78/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/128116204/0892ad32-42c3-4db5-9cbb-ae9dfb954f25)

Prediction

![image](https://github.com/Deeksha78/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/128116204/0f463b7b-5271-49a8-b851-b8b7688c27a9)

## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
