from PIL import Image
import numpy as np
import os
import cv2

# Code for making images into array -
data=[]
labels=[]
cats=os.listdir("cat")
for cat in cats:
    imag=cv2.imread("cat/"+cat)
    img_from_ar = Image.fromarray(imag, 'RGB')
    resized_image = img_from_ar.resize((50, 50))
    data.append(np.array(resized_image))
    labels.append(0)

dogs=os.listdir("dog")
for dog in dogs:
    imag=cv2.imread("dog/"+dog)
    img_from_ar = Image.fromarray(imag, 'RGB')
    resized_image = img_from_ar.resize((50, 50))
    data.append(np.array(resized_image))
    labels.append(1)
    
birds=os.listdir("bat")
for bird in birds:
    imag=cv2.imread("bat/"+bird)
    img_from_ar = Image.fromarray(imag, 'RGB')
    resized_image = img_from_ar.resize((50, 50))
    data.append(np.array(resized_image))
    labels.append(2)
    
fishes=os.listdir("goldfish")
for fish in fishes:
    imag=cv2.imread("goldfish/"+fish)
    img_from_ar = Image.fromarray(imag, 'RGB')
    resized_image = img_from_ar.resize((50, 50))
    data.append(np.array(resized_image))
    labels.append(3)

# Since the "data" and "labels" are normal array , convert them to numpy arrays-
animals=np.array(data)
labels=np.array(labels)

# Now save these numpy arrays so that you dont need to do this image manipulation again.

np.save("animals",animals)
np.save("labels",labels)

# Load the arrays ( Optional : Required only if you have closed your jupyter notebook after saving numpy array )

animals=np.load("animals.npy")
labels=np.load("labels.npy")

# Now shuffle the "animals" and "labels" set so that you get good mixture when you separate the dataset into train and test

s=np.arange(animals.shape[0])
np.random.shuffle(s)
animals=animals[s]
labels=labels[s]

# Make a variable num_classes which is the total number of animal categories and a variable data_length which is size of dataset

num_classes=len(np.unique(labels))
data_length=len(animals)

# Divide data into test and train

# Take 90% of data in train set and 10% in test set

(x_train,x_test)=animals[(int)(0.1*data_length):],animals[:(int)(0.1*data_length)]
x_train = x_train.astype('float32')/255
x_test = x_test.astype('float32')/255
train_length=len(x_train)
test_length=len(x_test)

# Divide labels into test and train

(y_train,y_test)=labels[(int)(0.1*data_length):],labels[:(int)(0.1*data_length)]

# Make labels into One Hot Encoding

import keras
from keras.utils import np_utils
#One hot encoding
y_train=keras.utils.to_categorical(y_train,num_classes)
y_test=keras.utils.to_categorical(y_test,num_classes)

# Making Keras Model

# import sequential model and all the required layers
from keras.models import Sequential
from keras.layers import Conv2D,MaxPooling2D,Dense,Flatten,Dropout

#make model
model=Sequential()
model.add(Conv2D(filters=16,kernel_size=2,padding="same",activation="relu",input_shape=(50,50,3)))
model.add(MaxPooling2D(pool_size=2))
model.add(Conv2D(filters=32,kernel_size=2,padding="same",activation="relu"))
model.add(MaxPooling2D(pool_size=2))
model.add(Conv2D(filters=64,kernel_size=2,padding="same",activation="relu"))
model.add(MaxPooling2D(pool_size=2))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(500,activation="relu"))
model.add(Dropout(0.2))
model.add(Dense(4,activation="softmax"))
model.summary()

# compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(x_train,y_train,batch_size=30,epochs=40,verbose=1)

# Test the model
score = model.evaluate(x_test, y_test, verbose=1)
print('\n', 'Test accuracy:', score[1])

def convert_to_array(img):
    im = cv2.imread(img)
    img = Image.fromarray(im, 'RGB')
    image = img.resize((50, 50))
    return np.array(image)
def get_animal_name(label):
    if label==0:
        return "cat"
    if label==1:
        return "dog"
    if label==2:
        return "bird"
    if label==3:
        return "fish"
def predict_animal(file):
    print("Predicting .................................")
    ar=convert_to_array(file)
    ar=ar/255
    label=1
    a=[]
    a.append(ar)
    a=np.array(a)
    score=model.predict(a,verbose=1)
    print(score)
    label_index=np.argmax(score)
    print(label_index)
    acc=np.max(score)
    animal=get_animal_name(label_index)
    print("The predicted Animal is a "+animal+" with accuracy =    "+str(acc))

predict_animal("1.jpg")
predict_animal("2.jpg")
predict_animal("3.jpg")
predict_animal("4.jpg")
predict_animal("5.jpg")

c = input("Enter to exit")
