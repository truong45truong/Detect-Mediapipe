from re import X
import numpy as np
import mediapipe as mp
import tensorflow as tf
from keras.layers import Dropout
from keras.models import Sequential
import cv2 
import os
import PIL 
import matplotlib.pyplot as plt
classes={'ac1':[1,0,0,0],'ac2':[0,1,0,0],'ac3':[0,0,1,0],'ac4':[0,0,0,1]}
TRAIN_DATA='./dataset_EX1/Traindata'
def read_File_Img(link):
    Img_list=[]
    for folder in os.listdir(link):
        folder_path=os.path.join(link,folder)
        list_filename_path=[]
        for filename in os.listdir(folder_path):
            filename_path=os.path.join(folder_path,filename)
            label=filename_path.split('\\')[1]
            img = cv2.imread(filename_path)
            list_filename_path.append((img,classes[label]))
        Img_list.extend(list_filename_path)
    return Img_list
def make_landmark_timestep(results):
    c_lm = []
    for id, lm in enumerate(results.pose_landmarks.landmark):
        c_lm.append(lm.x)
        c_lm.append(lm.y)
        c_lm.append(lm.z)
        c_lm.append(lm.visibility)
    return c_lm
def result_landmark(List_img):
    # Khởi tạo thư viện mediapipe
    mpPose = mp.solutions.pose
    pose = mpPose.Pose()
    lm_list = []
    for i in List_img:
        # Nhận diện pose
        frameRGB = cv2.cvtColor(i[0], cv2.COLOR_BGR2RGB)
        results = pose.process(frameRGB)
        if results.pose_landmarks:
            # Ghi nhận thông số khung xương
            lm = make_landmark_timestep(results)
            lm_list.append((lm,i[1]))
    return lm_list
X_train=result_landmark(read_File_Img(TRAIN_DATA))
#X_test=result_landmark(read_File_Img(TEST_DATA))
np.random.shuffle(X_train)
np.random.shuffle(X_train)
np.random.shuffle(X_train)
x_train=np.array([x[0] for i,x in enumerate(X_train)])
y_train=np.array([x[1] for i,x in enumerate(X_train)])
print(x_train[1].shape)
model_trainning_first = tf.keras.models.Sequential([
                                                    tf.keras.layers.Input(shape=(132,)),
                                                    tf.keras.layers.Dense(128),
                                                    tf.keras.layers.Dense(32),
                                                    tf.keras.layers.Dense(16),
                                                    tf.keras.layers.Dense(4, activation='softmax')
                                                    ])

model_trainning_first.summary()
model_trainning_first.compile(optimizer='adam',
                              loss='categorical_crossentropy',
                              metrics=['accuracy'])
print(x_train.shape)
model_trainning_first.fit(tf.expand_dims(x_train, axis=-1),y_train,epochs=50)
model_trainning_first.save('model_EX1.hdf5') 
