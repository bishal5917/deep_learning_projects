import os
import caer
import canaro
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import gc
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import LearningRateScheduler

IMG_SIZE = (80,80)
channels = 1
char_path = 'opencv_face_detection/simpsons_dataset'
BATCH_SIZE = 32
EPOCHS = 10
test_image_path = 'opencv_face_detection/simpsons_dataset/bart_simpson/pic_0010.jpg'

#we add all of these dataset folder into a dictionary
char_dict={}
for char in os.listdir(char_path):
    char_dict[char] =len(os.listdir(os.path.join(char_path,char)))

#sort in descending order
char_dict =caer.sort_dict(char_dict,descending=True)
# print(char_dict)

characters=[]
count = 0
for i in char_dict:
    characters.append(i[0])
    count+=1
    if count>=10:
        break
print(characters)

#creating the training data
train = caer.preprocess_from_dir(char_path,characters,channels=channels,IMG_SIZE=IMG_SIZE
        ,isShuffle=True)

#creating the featureset
featureSet,labels = caer.sep_train(train,IMG_SIZE=IMG_SIZE)

#normalise the feature set -> (0,1)
featureSet = caer.normalize(featureSet)
labels = to_categorical(labels,len(characters))

#creating the training set and the validation set (80-20)
x_train, x_val, y_train, y_val = caer.train_val_split(featureSet, labels, val_ratio= .2)

#Image data generator
#To induce some randomness to the images
datagen = canaro.generators.imageDataGenerator()
train_gen = datagen.flow(x_train,y_train,batch_size = BATCH_SIZE)

#creating the model
model = canaro.models.createSimpsonsModel(IMG_SIZE=IMG_SIZE, channels=channels, output_dim= len(characters),
                        loss='binary_crossentropy', learning_rate= 0.001, momentum=True,
                        nesterov=True)

#callbacks list
callbacks_list = [LearningRateScheduler(canaro.lr_schedule)]

#training the model
training = model.fit(
    train_gen,
    steps_per_epoch=len(x_train)//BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=(x_val,y_val),
    validation_steps=len(y_val)//BATCH_SIZE,
    callbacks=callbacks_list
)

img=cv.imread(test_image_path)
# plt.plot(img)

def prepare(img):
    img=cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    img=cv.resize(img,IMG_SIZE)
    img=caer.reshape(img,IMG_SIZE,1)
    return img

#giving one data and trying to find the answer
predictions=model.predict(prepare(img))

print(characters[np.argmax(predictions[0])])


