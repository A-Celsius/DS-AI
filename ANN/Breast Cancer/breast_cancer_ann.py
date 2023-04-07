import numpy as np
import pandas as pd

df = pd.read_csv('/Breast_cancer.csv')

df.drop(['Unnamed: 32','id'],axis=1,inplace=True)

x = df.drop('diagnosis',axis=1)
y = df.diagnosis

from sklearn.preprocessing import LabelEncoder
lb = LabelEncoder()
y = lb.fit_transform(y)

from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest = train_test_split(x,y,test_size=0.3,random_state=1)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
xtrain = scaler.fit_transform(xtrain)
xtest = scaler.transform(x)

import tensorflow
from tensorflow import keras
from keras import Sequential
from keras.layers import Dense

model = Sequential()

model.add(Dense(9,activation='relu',input_dim=30))
model.add(Dense(9,activation='relu'))
model.add(Dense(1,activation='sigmoid'))

model.summary()

model.compile(loss='binary_crossentropy',optimizer='Adam',metrics=['accuracy'])

model.fit(xtrain,ytrain,epochs=50)

y_pred=model.predict(xtest)
y_pred = (y_pred>0.5)
print(y_pred[:10])

from sklearn.metrics import confusion_matrix

from sklearn.metrics import accuracy_score

y_pred=np.argmax(y_pred,axis=1)

cm = confusion_matrix(ytest,y_pred)
score = accuracy_score(ytest,y_pred)
print(cm)
print("score is : ",score)

