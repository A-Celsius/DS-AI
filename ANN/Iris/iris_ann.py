import pandas as pd
import numpy as np

df = pd.read_csv("/content/sample_data/iris.csv")

x = df.drop('variety',axis=1)
y = df.variety

from sklearn.preprocessing import LabelEncoder
lb = LabelEncoder()
y = lb.fit_transform(df.variety)

df["species"] = y

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3)

print(x_train.shape)
print(x_test.shape)

import tensorflow
from tensorflow import keras
from keras import Sequential
from keras.layers import Dense

model = Sequential()
model.add(Dense(10,input_shape=(4,),activation='relu'))
model.add(Dense(8,activation='relu'))
model.add(Dense(3,activation='softmax'))

model.summary()

model.compile(optimizer="adam",loss="sparse_categorical_crossentropy", metrics=['accuracy'])

model.fit(x_train,y_train,epochs=100)

model.evaluate(x_test,y_test)

y_pred = model.predict(x_test)

y_pred=np.argmax(y_pred,axis=1)
print(y_pred)

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
cm = confusion_matrix(y_test,y_pred)
score = accuracy_score(y_test,y_pred)
print(cm)
print("score is : ",score)