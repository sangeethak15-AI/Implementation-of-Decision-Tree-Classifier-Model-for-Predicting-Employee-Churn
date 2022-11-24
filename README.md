# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required libraries.

2.Upload and read the dataset

3.Check for any missing value in the dataset using isnull function.

4.From sklearn.tree import DecisionTreeClassifier and use criteria as entropy

5.Find the accuracy of the model and predict the required values by importing the required modules from sklearn. 

## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: Sangeetha.K
RegisterNumber: 212221230085 
*/
```

```
import pandas as pd
data=pd.read_csv("Employee.csv")
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

![200617343-4b606eaf-6918-46b1-a7ef-91a49bf4e4fc](https://user-images.githubusercontent.com/93992063/203684371-b01eb569-a400-447b-966d-02ce9cbe9c35.png)

![ml62](https://user-images.githubusercontent.com/93992063/198622875-45307d86-74c2-4e6c-a424-18d1a5a6bb67.png)
![200617478-f81bd031-ccdd-4d22-8295-758384740aa5](https://user-images.githubusercontent.com/93992063/203684412-3880622e-6e65-4a0b-8070-801863478cfe.png)

![200617600-5af99405-5076-49df-b9e7-e3ea8696d097](https://user-images.githubusercontent.com/93992063/203684430-368b215c-1def-4b26-ac70-f70e7967c344.png)


![ml64](https://user-images.githubusercontent.com/93992063/198623007-6c66644a-90f8-46e1-9714-e9cae680a6eb.png)



![ml65](https://user-images.githubusercontent.com/93992063/198623098-83e674c9-770c-40c2-b34f-578a4a43d000.png)

![200617788-622864f2-ae41-4327-b5c9-776ebcad87b7](https://user-images.githubusercontent.com/93992063/203684446-6bb047ad-f5de-4bbc-8679-de43ec254ba6.png)

![200617741-207d4daa-627a-4ddd-99b0-4bdfd63d5365](https://user-images.githubusercontent.com/93992063/203684455-305f9f97-7f18-4d47-9350-57052298c757.png)

![200617830-127c32a2-fb2a-48b7-87ad-391705a0f589](https://user-images.githubusercontent.com/93992063/203684469-96b56a89-ea6d-4c65-a2a7-6d5d4cff328b.png)


## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
