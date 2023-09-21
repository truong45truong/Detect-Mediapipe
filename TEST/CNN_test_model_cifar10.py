import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


(X_train,y_train),(X_test,y_test)=tf.keras.datasets.cifar10.load_data()
X_train=X_train/255
X_test=X_test/255
y_train,y_test= tf.keras.utils.to_categorical(y_train),tf.keras.utils.to_categorical(y_test),
classes=['airplane','automobile','bird','cat','dear','dog','frog','hourse','ship','truck']

models=tf.keras.models.load_model('model-cifar10.h5')

for i in range(50):
    plt.subplot(5,10,i+1)
    plt.imshow(X_test[i])
    plt.title(classes[np.argmax(models.predict(X_test[i].reshape((-1,32,32,3))))])
    plt.axis('off')
plt.show()
