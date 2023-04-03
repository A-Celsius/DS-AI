import pandas as pd

churn = pd.read_csv("/content/Churn_Modelling.csv")

churn.head()

churn.info()

churn.Gender.unique()

churn.Gender = churn.Gender.map({"Male":0,"Female":1})

churn.drop({"Surname","Geography",'RowNumber','CustomerId'},inplace=True,axis=1)

churn.columns

features = churn.drop("Exited",axis=1)

target = churn.Exited

features.head()

features.info()

target.head()



from sklearn.model_selection import train_test_split

X_train,X_test,Y_train,Y_test = train_test_split(features,target,test_size=0.3)

X_train.info()

X_train.shape

X_test.info()





import tensorflow
from tensorflow import keras
from keras.layers import Dense,Flatten
from keras.models import Sequential

tensorflow.__version__

model = Sequential()



model.add(Dense(10,input_dim=(9),activation="relu"))
model.add(Dense(5,activation='relu'))
model.add(Dense(1,activation='sigmoid'))

model.summary()

model.compile(loss='binary_crossentropy',optimizer='Adam',metrics=['accuracy'])

model.fit(X_train,Y_train,epochs=10)

