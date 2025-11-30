# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Load the dataset, drop unnecessary columns, and encode categorical variables. 
2.Define the features (X) and target variable (y). 
3.Split the data into training and testing sets. 
4.Train the logistic regression model, make predictions, and evaluate using accuracy and other 
 
## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

data = pd.read_csv(r"C:\Users\admin\Desktop\Placement_Data.csv")
data.head()

datal = data.copy()
datal = datal.drop(["sl_no", "salary"], axis=1)
datal.head()

print("Missing values:\n", datal.isnull().sum())
print("Duplicate rows:", datal.duplicated().sum())

le = LabelEncoder()
datal["gender"] = le.fit_transform(datal["gender"])
datal["ssc_b"] = le.fit_transform(datal["ssc_b"])
datal["hsc_b"] = le.fit_transform(datal["hsc_b"])
datal["hsc_s"] = le.fit_transform(datal["hsc_s"])
datal["degree_t"] = le.fit_transform(datal["degree_t"])
datal["workex"] = le.fit_transform(datal["workex"])
datal["specialisation"] = le.fit_transform(datal["specialisation"])
datal["status"] = le.fit_transform(datal["status"])

x = datal.iloc[:, :-1]
y = datal["status"]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

lr = LogisticRegression(solver="liblinear")
lr.fit(x_train, y_train)

y_pred = lr.predict(x_test)

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

confusion = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", confusion)

classification_report_output = classification_report(y_test, y_pred)
print("Classification Report:\n", classification_report_output)
Developed by:Girishva.K
RegisterNumber:25009292  
*/
```

## Output:
HEAD
<img width="1279" height="330" alt="384694261-6d3166e2-7fd9-4582-8729-4ba68a64036b" src="https://github.com/user-attachments/assets/b48556a1-e11f-4d69-9eee-b2a493dd64a4" />
COPY
<img width="1141" height="345" alt="384694385-3e4ee2fa-f457-4c2c-a13a-d4791707c7d7" src="https://github.com/user-attachments/assets/8872d76b-9c5c-489f-9de4-75bb1a763028" />
FIT TRANSFORM
<img width="1122" height="707" alt="384694436-b0e4a287-35b1-4c2d-b386-d20f8f62b772" src="https://github.com/user-attachments/assets/8b8982fa-cedc-426f-9280-f8aab6765f4a" />
LOGISTIC REGRESSION
<img width="1231" height="309" alt="384694500-49664d5e-2913-456b-95fa-dd7ae5a15637" src="https://github.com/user-attachments/assets/5fe4d63f-3a88-4e60-9a7c-0cb9d01c92ef" />
ACCURACY SCORE
<img width="1225" height="169" alt="384694533-0e91c61e-abcb-400f-a04f-c36ec2885d1c" src="https://github.com/user-attachments/assets/96646771-1265-4b16-ab5b-634b6bb4dbf7" />
CONFUSION MATRIX
<img width="1229" height="203" alt="388782587-58ce1bb2-1678-4407-9b13-cf33fabb3cc9" src="https://github.com/user-attachments/assets/be619a78-99d5-4273-8ac9-76c7110fbe3d" />
CLASSIFICATION REPORT & PREDICTION
<img width="1217" height="524" alt="384694700-f1591a15-9289-4a4e-a284-20d5b58298b3" src="https://github.com/user-attachments/assets/51e07612-18ec-46fc-bb35-27584f3fb445" />



## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
