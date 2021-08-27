import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator


train_datagen = ImageDataGenerator(
    rescale= 1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

train_set =  train_datagen.flow_from_directory(
        'dataset/training_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')

test_datagen = ImageDataGenerator(rescale=1./255)
test_set = test_datagen.flow_from_directory(
        'dataset/test_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')


cnn = tf.keras.models.Sequential()

# convolution layer 1
cnn.add(tf.keras.layers.Conv2D(filters=32,kernel_size = 3,activation = 'relu', input_shape = [64,64,3]))
# max pooling layer 1
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides = 2))
# convolution layer 2
cnn.add(tf.keras.layers.Conv2D(filters=32,kernel_size = 3,activation = 'relu'))
# max pooling layer 2
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides = 2))
# flattening
cnn.add(tf.keras.layers.Flatten())

cnn.add(tf.keras.layers.Dense(units=128, activation='relu'))

cnn.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

cnn.compile(optimizer ='adam', loss ='binary_crossentropy', metrics = ['accuracy'])

cnn.fit(x= train_set, validation_data = test_set, epochs = 25)


# making single prediction
import numpy as np
from keras.preprocessing import image 
test_image = image.load_img("dataset/single_prediction/cat_or_dog_1.jpg", target_size=(64,64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)
res = cnn.predict(test_image)
print(train_set.class_indices)
if res[0][0] == 1:
        pred = "dog"
else:
        pred = "cat"
print(pred)

# saving model to json file
model_json = cnn.to_json()
with open("model.json","w")  as json_file:
        json_file.write(model_json)

cnn.save_weights("model.h5")



