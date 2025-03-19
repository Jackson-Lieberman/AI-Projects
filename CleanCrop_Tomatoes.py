
from keras.models import load_model, Sequential
from keras.layers import Conv2D, Input, MaxPooling2D, Activation, Dropout, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array

from PIL import Image
import numpy as np
import os

Healthy = 0                                                              #creates lists
Yellow_Leaf_Curl_Virus= 0                                                         
Bacterial_spot = 0
Early_blight = 0
Late_blight = 0
Leaf_Mold = 0
Septoria_leaf_spot = 0
Spider_mites  = 0 
Target_Spot = 0
Tomato_mosaic_virus =0 



img_width, img_height = 250, 250                                         #set image width and height

trainRescale = ImageDataGenerator(rescale=1./255)                        #Normalizes the pixel values of the images from a range of 0-255 to a range of 0-1

trainData = trainRescale.flow_from_directory(                            #takes and labels the normalized images based on the folder
    'Tomatoes/train.tomatoes/',                                                        #folder it is being trained on
    target_size = (img_width,img_height),                                #specifies dimentions for input images
    batch_size = 32,                                                     #number of images to precess in a single batch of training
    class_mode = "categorical")                                          #class labels are processed in categories

model = Sequential()

model.add(Input(shape=(img_height, img_width, 3)))                       #adds a layer with 32 filters and a kernel shape of 3x3 (3 channels rgb)
model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))                                            #adds relu activatation that replaces - with 0
model.add(MaxPooling2D(pool_size = (2, 2)))                              #reduces image and take max values from a 2x2 pixel region for simplification
#another layer (repeats)
model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu')) 
model.add(MaxPooling2D(pool_size = (2, 2)))
#same again just 64 pixel filter
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu')) 
model.add(MaxPooling2D(pool_size = (2, 2)))

model.add(Flatten())                                                     #flattens output into 1d vector
model.add(Dense(64))                                                     #passes the vectors to a layer with 64 nurons
model.add(Activation('relu')) 
model.add(Dropout(0.5))                                                  #prevents over fitting, drops 50% of data

num_classes = 10                                                          # Number of classes (Healthy, DownyMildew, PowderMildew, LeafMiner)
model.add(Dense(num_classes))
model.add(Activation("softmax"))                                         #produces value as softmax for multi-class classification

model.compile(loss='categorical_crossentropy',                           #compiles model
              optimizer = "rmsprop",
              metrics = ['accuracy'])

model.fit(                                                               #trains the model 
    trainData,                                                           #what the model is trained on
    steps_per_epoch =12,                                                 #updates 12 times per epoch
    epochs = 128)                                                        #128 epochs

model.save_weights('CleanCrop-Tomato.weights.h5')                      #saves weights
model.save('CleanCrop-Tomato.keras')                             #saves model

testImages = os.listdir('Tomatoes/test.tomatoes/')                                         #gets list of all images in test folder
class_labels = {0: "Healthy tomato leaf", 1: "leaf with yellow leaf curl virus", 2: "leaf with a bacterial spot", 3: "leaf with early blight", 4: "leaf with late blight", 5: "leaf with leaf mold", 6: "leaf with septoria leaf spot", 7: "leaf with spider mites", 8: "leaf with target spot", 9: "leaf with tomato mosaic virus"}  #makes labels for each class

for image in testImages:                                                 #loops through images
    try:
        img = Image.open('Tomatoes/test.tomatoes/'+ image)                                 #opens image
        img = img.resize((img_width, img_height))                        #resize to 150x150
        img = img_to_array(img)                                          #makes the img an array
        img = np.expand_dims(img, axis = 0)                              # adds demension to img
        
        result = model.predict(img)                                      #predicts out put

        predicted_class = np.argmax(result)                              # Get index of highest probability
        prediction = class_labels[predicted_class]                       # Get the label for that class
        print('The image', image, 'is a:', prediction)                   #prints the prediction
        if prediction == "Healthy tomato leaf":                                 #adds to the values of each leaf if thats what they are predicted as
            Healthy += 1
        elif prediction == "leaf with yellow leaf curl virus":
            Yellow_Leaf_Curl_Virus += 1
        elif prediction == "leaf with a bacterial spot":
            Bacterial_spot += 1
        elif prediction == "leaf with early blight":
            Early_blight += 1
        elif prediction == "leaf with late blight":
            Late_blight += 1
        elif prediction == "leaf with leaf mold":
            Leaf_Mold += 1
        elif prediction == "leaf with septoria leaf spot":
            Septoria_leaf_spot += 1
        elif prediction == "leaf with spider mites":
            Spider_mites += 1
        elif prediction == "leaf with target spot":
            Target_Spot += 1
        elif prediction == "leaf with tomato mosaic virus":
            Tomato_mosaic_virus += 1
    except Exception as e:                                               #if it cant read the image it will print tha
        print(f"Error processing {image}: {e}")
print("Healthy leaves:", Healthy)                                        #prints how many leaves had each case
print("Yellow leaf curl virus leaves:", Yellow_Leaf_Curl_Virus)
print("Bacterial spot leaves:", Bacterial_spot)
print("Ealy blight leaves:", Early_blight)
print("Late blight leaves:", Late_blight)
print("Leaf mold leaves:", Leaf_Mold)
print("Septoria leaf spot leaves:", Septoria_leaf_spot)
print("Spider mites leaves:", Spider_mites)                                        
print("Target spot leaves:", Target_Spot)
print("Tomato mosaic virus leaves:", Tomato_mosaic_virus)
