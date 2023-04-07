import pandas as pd

churn = pd.read_csv("/content/Churn_Modelling.csv")

churn.Gender = churn.Gender.map({"Male":0,"Female":1})

churn.drop({"Surname","Geography",'RowNumber','CustomerId'},inplace=True,axis=1)

features = churn.drop("Exited",axis=1)

target = churn.Exited


from sklearn.model_selection import train_test_split

X_train,X_test,Y_train,Y_test = train_test_split(features,target,test_size=0.3)


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

import tensorflow
from tensorflow import keras
from keras.layers import Dense,Flatten
from keras.models import Sequential

model = Sequential()



model.add(Dense(7,input_dim=(9),activation="relu"))
model.add(Dense(5,activation='relu'))
model.add(Dense(3,activation='relu'))
model.add(Dense(1,activation='sigmoid'))

model.summary()

model.compile(loss='binary_crossentropy',optimizer='Adam',metrics=['accuracy'])

model.fit(X_train,Y_train,epochs=50)

