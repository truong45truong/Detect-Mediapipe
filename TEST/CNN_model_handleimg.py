import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

(X_train,y_train),(X_test,y_test) =tf.keras.datasets.cifar10.load_data()
X_train=X_train/255
X_test=X_test/255
y_train,y_test= tf.keras.utils.to_categorical(y_train),tf.keras.utils.to_categorical(y_test),
classes=['airplane','automobile','bird','cat','dear','dog','frog','hourse','ship','truck']

model_trainning_first = tf.keras.models.Sequential([tf.keras.layers.Conv2D(32,(3,3),input_shape=(32,32,3),activation='relu'),
                                                    tf.keras.layers.MaxPool2D((2,2)),
                                                    tf.keras.layers.Dropout(0.15),

                                                    tf.keras.layers.Conv2D(64,(3,3),activation='relu'),
                                                    tf.keras.layers.MaxPool2D((2,2)),
                                                    tf.keras.layers.Dropout(0.15),

                                                    tf.keras.layers.Conv2D(128,(3,3),activation='relu'),
                                                    tf.keras.layers.MaxPool2D((2,2)),
                                                    tf.keras.layers.Dropout(0.2),

                                                    tf.keras.layers.Flatten(input_shape=(32,32,3)),
                                                    tf.keras.layers.Dense(1000, activation='relu'),
                                                    tf.keras.layers.Dense(256,activation='relu'),
                                                    tf.keras.layers.Dense(10, activation='softmax')
                                                    ])

model_trainning_first.summary()
model_trainning_first.compile(optimizer='adam',
                              loss='categorical_crossentropy',
                              metrics=['accuracy'])
model_trainning_first.fit(X_train,y_train,epochs=1)
print(X_train.shape,X_train[1].shape)
model_trainning_first.save('model-cifar10.h5') 
