import numpy as np # type: ignore
import pandas as pd #type: ignore
import joblib as jb
from sklearn.ensemble import RandomForestClassifier #type:ignore
from sklearn.preprocessing import LabelEncoder   #type:ignore
from sklearn.preprocessing import StandardScaler #type:ignore 
from sklearn.model_selection import train_test_split #type:ignore
from sklearn.metrics import accuracy_score #type:ignore

df=pd.read_csv('Training.csv')
X=df.iloc[:,0:132]
Y=df['prognosis']
le=LabelEncoder()
Y=le.fit_transform(Y)
ss=StandardScaler()
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.6,random_state=42)
model=RandomForestClassifier(max_depth=10,min_samples_split=10,min_samples_leaf=5)
model.fit(X_train,Y_train)
Y_pred=model.predict(X_test)
max=0
for i in range(1,546):
    accuracy=accuracy_score(Y_pred,Y_test)
    if(accuracy>=max):
        max=accuracy
if(max>=0.93):
    print("The accuracy of the model is ",max)
    jb.dump(model,'diseasepred.pkl')
    print("Model saved succesfully")
else:
    print("The accuracy of the model is ",max)
    print("Model not saved")