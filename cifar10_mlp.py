import pickle
import numpy as np
import matplotlib.pyplot as plt
#from random import random
import random
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def unpickle(file):
    with open(file, 'rb') as f:
        dict = pickle.load(f, encoding="latin1")
    return dict

labeldict = unpickle('C:/Users/zeyne/Desktop/ex5/cifar-10-batches-py/batches.meta')
label_names = labeldict["label_names"]

x_train = []
y_train = [] 
x_test = []
y_test=[]

for i in range(1,6):
    datadict = unpickle(f'C:/Users/zeyne/Desktop/ex5/cifar-10-batches-py/data_batch_{i}')
    x_train.append(datadict["data"])
    y_train.extend(datadict["labels"])

datadict = unpickle('C:/Users/zeyne/Desktop/ex5/cifar-10-batches-py/test_batch')
x_test = datadict["data"].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1).astype("uint8")
y_test = datadict["labels"]

num_classes = 10
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

x_train = np.concatenate(x_train, axis=0)
x_train = x_train.reshape(-1,3,32,32).transpose(0,2,3,1).astype("uint8")

x_train, x_test = x_train/255.0, x_test/255.0


model = keras.Sequential([
    keras.layers.Flatten(input_shape=(32, 32, 3)),  
    keras.layers.Dense(50, activation='relu'),      
    keras.layers.Dense(30, activation='sigmoid'),    #After some trainings, I added hidden layer consisting of 30 neurons and also changed the activation function here to sigmoid after trying others
    keras.layers.Dense(num_classes, activation='softmax')
])

model.compile(optimizer='Adam', #I also manipulated the learning rate but it didn't affect the training well so I decided on not manipulating it
              loss='categorical_crossentropy',
              metrics=['accuracy'])


#I first started with 10 number of epochs but it wasn't enough so I also trained the model with 15 and 30 epochs
history=model.fit(x_train, y_train, epochs=30, batch_size=64, validation_split=0.2)
plt.plot(history.history['loss'], label='Training Loss') #Plotting the training loss graphic
plt.xlabel('Epoch')
plt.ylabel('Loss')           
plt.legend()
plt.show()


train_loss,train_accuracy = model.evaluate(x_train,y_train)
print(f"Training Accuracy: {train_accuracy * 100:.2f}%")
test_loss, test_accuracy = model.evaluate(x_test, y_test)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

#prediction
index = random.randint(0,10000)
predictions = model.predict(x_test)
label = np.argmax(predictions[index])
plt.imshow(x_test[index])
plt.title(f"Predicted Class: {label}")
plt.show()