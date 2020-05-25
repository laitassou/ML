#BASIC
import numpy as np 
import pandas as pd 
import os

# DATA visualization
import matplotlib.pyplot as plt
import seaborn as sns
import PIL
from IPython.display import Image, display
from plotly import graph_objs as go
import plotly.express as px
import plotly.figure_factory as ff

import openslide


from PIL import Image



import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
import keras.utils
from keras import utils as np_utils

from sklearn.model_selection import train_test_split



from tensorflow.python.client import device_lib 




from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions


import  tensorflow.keras.optimizers 

import math

#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
print(device_lib.list_local_devices())

show= 0

import sys, getopt



BASE_FOLDER = "/datadrive/"
##!ls {BASE_FOLDER}

mask_dir = f'{BASE_FOLDER}/train_label_masks'


extracted_mask_dir = f'{BASE_FOLDER}/extracted_masks'
extracted_train_dir = f'{BASE_FOLDER}/extracted_train'


train = pd.read_csv(BASE_FOLDER+"train.csv")
test = pd.read_csv(BASE_FOLDER+"test.csv")
sub = pd.read_csv(BASE_FOLDER+"sample_submission.csv")

print(train.head())


print("unique ids : ", len(train.image_id.unique()))
print("unique data provider : ", len(train.data_provider.unique()))
print("unique isup_grade(target) : ", len(train.isup_grade.unique()))
print("unique gleason_score : ", len(train.gleason_score.unique()))


train['gleason_score'].unique()



print(train[train['gleason_score']=='0+0']['isup_grade'].unique())
print(train[train['gleason_score']=='negative']['isup_grade'].unique())



print(len(train[train['gleason_score']=='0+0']['isup_grade']))
print(len(train[train['gleason_score']=='negative']['isup_grade']))

print(train[(train['gleason_score']=='3+4') | (train['gleason_score']=='4+3')]['isup_grade'].unique())
print(train[(train['gleason_score']=='3+5') | (train['gleason_score']=='5+3')]['isup_grade'].unique())
print(train[(train['gleason_score']=='5+4') | (train['gleason_score']=='4+5')]['isup_grade'].unique())



print(train[train['gleason_score']=='3+4']['isup_grade'].unique())
print(train[train['gleason_score']=='4+3']['isup_grade'].unique())



print(train[(train['isup_grade'] == 2) & (train['gleason_score'] == '4+3')])



train.drop([7273],inplace=True)

train['gleason_score'] = train['gleason_score'].apply(lambda x: "0+0" if x=="negative" else x)




  
# dropping passed columns 
train.drop(["data_provider", "isup_grade"], axis = 1, inplace = True) 

print("before")
print(train.head())


temp = train.groupby('gleason_score').count()['image_id'].reset_index().sort_values(by='image_id',ascending=False)
temp.style.background_gradient(cmap='Reds')

print(temp)


train['gleason_score'] = train['gleason_score'].apply(lambda x: "0" if x=="0+0" else (     \
"1" if x=="3+3" else ( "2" if x=="3+4" else ("3" if x=="4+3" else ("4" if x=="4+4" else (\
"5" if x=="4+5" else ("6") if x=="5+4" else ("7" if x=="5+5" else ("8" if x=="3+5" else ("9" if x=="5+3" else "-1")))
))))) )

print("after")

print(train.head())

temp = train.groupby('gleason_score').count()['image_id'].reset_index().sort_values(by='image_id',ascending=False)
temp.style.background_gradient(cmap='Reds')

print(temp)


print("rows+" )
print(train.shape)


data_array = []
train_value = []


dimension = 16
nb_lines = int(math.sqrt(dimension))

lines = nb_lines*128




start=0
def main(argv):
   outputfile = ''
   try:
      opts, args = getopt.getopt(argv,"hi:o:",["ifile=","ofile="])
   except getopt.GetoptError:
      sys.exit(2)
   for opt, arg in opts:
      if opt == '-h':
         sys.exit()
      elif opt in ("-i", "--ifile"):
         start = arg
         print(start)
      elif opt in ("-o", "--ofile"):
         outputfile = arg


BATCH_IMAGES = 700
def compute () :
   print("lines:"+str(lines))
   j=0
   for index, row in train.iterrows():
      images_id = row['image_id']
      score = row['gleason_score']
      #print(images_id+ " "+score)
      i=0

      if j <start :
         pass
      if j > start+ BATCH_IMAGES:
         break
      j= j+1

      slice_array = []
      for k in range (0,dimension):
         try:
               img = Image.open(f'{extracted_train_dir}/{images_id}_{k}.png').convert('RGBA')
               arr = np.array(img)
               #print(arr.shape)
               r, g, b, a = np.rollaxis(arr, axis=-1)
               #R =arr.take(0,axis=2)
               #G = arr.take(1 ,axis=2)
               #B = arr.tak(2,axis=2)i
               x = np.dstack([r,g,b])
               R_ = arr # np.array(R)
               print('rgb,'+ str(x.shape))
               if k == 0:
                  slice_array = x #R_
               else:
                  #print ("test:" + str(k))    
                  #print(R)
                  slice_array= np.concatenate((slice_array, x ),axis=1)
                  print(slice_array.shape)

               #if k ==0:
               #    train_value.append(score)

               if k == dimension-1:
                  #print ("test")
                  data_array.append(slice_array.reshape(lines,lines,3))
                  train_value.append(score)
               img.close()



         except:
               pass 
      #if j ==700:
      #   break



   mat_data = np.array(data_array)
   print('mat_data:')
   print(mat_data.shape)
   print(mat_data.shape[0])


   mat_train = np.array(train_value)
   print('mat_train:')
   print(mat_train.shape)





   X = mat_data.reshape(mat_data.shape[0], lines, lines, 3)
   #x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
   input_shape = (lines, lines, 3)
   # convert class vectors to binary class matrices
   Y = keras.utils.to_categorical(mat_train, num_classes=10)
   #y_test = keras.utils.to_categorical(y_test, num_classes)
   X = X.astype('float32')
   #x_test = x_test.astype('float32')
   X /= 255
   #x_test /= 255
   print('x_train shape:', X.shape)
   print(X.shape[0], 'train samples')
   #print(x_test.shape[0], 'test samples')




   X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

   batch_size = 1
   num_classes = 10
   epochs = 30
   print("model")

   import keras 
   import tensorflow as tf
   from keras_applications.resnext import ResNeXt50
   from keras_applications.resnext import preprocess_input as resnext50_preprocess_input
   from keras.callbacks import ModelCheckpoint
   #from keras.applications.resnext import ResNeXt50


   import os.path

   filepath = "model.h5"
   file_exists = os.path.exists(file_path)
   if not (file_exists):
      #resnet50_imagenet_model = ResNeXt50(include_top=False, weights='imagenet' , input_shape=(lines, lines, 3))
      resnet50_imagenet_model = ResNeXt50(include_top=False, weights='imagenet' , input_shape=(lines, lines, 3),backend=keras.backend,layers = keras.layers, models = keras.models, utils = keras.utils )

      #Flatten output layer of Resnet
      flattened = keras.layers.Flatten()(resnet50_imagenet_model.output)

      #Fully connected layer 1
      fc1 = keras.layers.Dense(128, activation='relu', name="AddedDense1")(flattened)

      #Fully connected layer, output layer
      fc2 = keras.layers.Dense(10, activation='softmax', name="AddedDense2")(fc1)

      model = keras.models.Model(inputs=resnet50_imagenet_model.input,outputs=fc2)


      model.summary()

      model.compile(loss=keras.losses.categorical_crossentropy,optimizer=tf.keras.optimizers.Adadelta(),metrics=['accuracy'])


   else:
      model = load_model(filepath)
      model.summary()

   checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
   callbacks_list = [checkpoint]

   # fit the model
   #model.fit(x_train, y_train, epochs=5, batch_size=50, callbacks=callbacks_list)

   # load the model
   #new_model = load_model(filepath)
   #assert_allclose(model.predict(x_train),   new_model.predict(x_train), 1e-5)

   # fit the model
   #checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
   #callbacks_list = [checkpoint]
   #new_model.fit(x_train, y_train, epochs=5, batch_size=50, callbacks=callbacks_list)



   hist = model.fit(X_train, y_train,batch_size=batch_size,epochs=epochs,verbose=1,validation_data=(X_test, y_test),callbacks=callbacks_list)
   print("The model has successfully trained")
   model.save('mnist.h5')
   print("Saving the model as mnist.h5")

   print("end")


   score = model.evaluate(X_test, y_test, verbose=0)
   print('Test loss:', score[0])
   print('Test accuracy:', score[1])





if __name__ == "__main__":
   main(sys.argv[1:])
   compute()
