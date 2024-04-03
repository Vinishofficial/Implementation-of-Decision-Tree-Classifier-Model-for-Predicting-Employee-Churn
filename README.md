# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import pandas
2. Import Decision tree classifier
3. Fit the data in the model
4. Find the accuracy score

## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: Raja Lakshmi E
RegisterNumber: 212222220033 
*/
```
```
import pandas as pd
data=pd.read_csv("/content/Employee.csv")
data.head()
data.info()
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
## data.head()
![Screenshot 2024-04-03 113240](https://github.com/Vinishofficial/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/146931793/b3ba5629-c51f-4b65-ba02-37437adb452f)

## data.info()
![Screenshot 2024-04-03 113058](https://github.com/Vinishofficial/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/146931793/b6ff53c8-9566-4ba8-b039-20ffcba32f78)

## data.isnull().sum()
![Screenshot 2024-04-03 113105](https://github.com/Vinishofficial/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/146931793/0fb12213-5611-463d-90f4-e50a1cf4885a)

## data value count
![Screenshot 2024-04-03 113110](https://github.com/Vinishofficial/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/146931793/f6db435f-8b50-4627-abfe-2ef132154d1d)

## data.head() for salary
![Screenshot 2024-04-03 113240](https://github.com/Vinishofficial/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/146931793/a929098a-8fe2-4453-8b52-69363d8f1057)

## x.head()
![Screenshot 2024-04-03 113240](https://github.com/Vinishofficial/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/146931793/b8936b9c-d593-4cce-8461-d34f88d7f995)

## accuracy value
![Screenshot 2024-04-03 113115](https://github.com/Vinishofficial/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/146931793/4f8b5de5-7211-4c51-8b24-b19d897bba02)

## data prediction
![Screenshot 2024-04-03 113131](https://github.com/Vinishofficial/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/146931793/d86e346b-3bb6-4058-9fbc-8cef5666f852)












## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
