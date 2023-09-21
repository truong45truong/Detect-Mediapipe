import tensorflow


model=tensorflow.keras.models.load_model("./model_Yoga.hdf5")

print(model.predic())