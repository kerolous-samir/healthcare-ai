import pandas as pd
import numpy as np
import seaborn as sns
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from skimage import io
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras import backend as K
import tensorflow.keras.saving
import random


brain_df = pd.read_csv("data\\data_mask.csv")
brain_df["mask"].value_counts()

fig = go.Figure([go.Bar(x=brain_df['mask'].value_counts().index , y=brain_df['mask'].value_counts())])
fig.update_traces(marker_color = 'rgb(200,0,0)', marker_line_color = 'rgb(0,255,0)',marker_line_width = 3, opacity = 0.6)
fig.show()

plt.imshow(cv2.imread(brain_df.mask_path[622]))
plt.show()
plt.imshow(cv2.imread(brain_df.image_path[622]))
plt.show()
cv2.imread(brain_df.image_path[622])


fig , axs = plt.subplots(12,3, figsize = (20,50))
count = 0
for i in range(len(brain_df)):
    if brain_df['mask'][i] == 1 and count <12:
        img = io.imread(brain_df.image_path[i])
        axs[count][0].title.set_text('Brain MRI')
        axs[count][0].imshow(img)
        
        mask = io.imread(brain_df.mask_path[i])
        axs[count][1].title.set_text('Brain Mask')
        axs[count][1].imshow(mask, cmap= 'grey')
        
        img[mask == 255] = (255,0,0)
        axs[count][2].title.set_text('Brain MRI with Mask')
        axs[count][2].imshow(img)
        count += 1
        
fig.tight_layout()

brain_df.drop(columns=['patient_id'],inplace=True)

brain_df['mask'] = brain_df['mask'].apply(lambda x: str(x))
from sklearn.model_selection import train_test_split

train , test = train_test_split(brain_df , test_size=0.15)

from tensorflow.keras.preprocessing.image import ImageDataGenerator 

datagen = ImageDataGenerator(rescale= 1./255. , validation_split=0.15)
testgen = ImageDataGenerator(rescale=1./255.)

datagenerator = datagen.flow_from_dataframe(train, directory='./',
                                           x_col='mask_path',y_col='mask',class_mode='categorical',
                                            batch_size=16,subset='training',target_size=(256, 256),shuffle=True)

valgenerator = datagen.flow_from_dataframe(train, directory='./',
                                           x_col='mask_path',y_col='mask',class_mode='categorical',
                                            batch_size=16,subset='validation',target_size=(256, 256),shuffle=True)

testgenerator = testgen.flow_from_dataframe(test, directory='./',
                                           x_col='mask_path',y_col='mask',class_mode='categorical',
                                            batch_size=16,target_size=(256, 256),shuffle=False)

from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.layers import Input

base_model = ResNet50(include_top=False,weights='imagenet',input_tensor=Input(shape=(256,256,3)))

for layer in base_model.layers:
    layer.trainable = False
    
from tensorflow.keras.layers import AveragePooling2D , Flatten , Dense ,Dropout
from tensorflow.keras.models import Model
heading = base_model.output
heading = AveragePooling2D(pool_size=(4,4))(heading)
heading = Flatten(name='flatten')(heading)
heading = Dense(256,activation='relu')(heading)
heading = Dropout(0.3)(heading)
heading = Dense(256,activation='relu')(heading)
heading = Dropout(0.3)(heading)
heading = Dense(2,activation='softmax')(heading)


model = Model(inputs = base_model.input , outputs =heading )

from tensorflow.keras.callbacks import EarlyStopping , ModelCheckpoint
earlystopping = EarlyStopping(monitor='val_loss',mode='min',verbose=1,patience=25)
modelcheckpoint = ModelCheckpoint(filepath="model_checkpoint.keras" ,monitor='val_loss',
    verbose=1,save_best_only=True)

model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
history = model.fit(datagenerator,steps_per_epoch=datagenerator.n //16,epochs=1,validation_data=valgenerator,validation_steps=valgenerator.n //16,callbacks=[earlystopping,modelcheckpoint])
model_json = model.to_json()
with open('test.json','w') as test2:
    trained_model = test2.write(model_json)
with open('test.json','r') as json_file:
    bestmodel = json_file.read()
from tensorflow.keras.models import model_from_json
model_best = model_from_json(bestmodel)
model_best.load_weights(".weights.h5")
model.save_weights(".weights.h5",overwrite=True)

model_best.compile(loss="categorical_crossentropy",optimizer='adam',metrics=['accuracy'])

test_predict = model_best.predict(testgenerator,steps=testgenerator.n //16,verbose=1)

prediction = []
for i in test_predict:
    prediction.append(str(np.argmax(i)))
prediction  = np.asarray(prediction)


original = np.asarray(test['mask'])[:len(prediction)]

from sklearn.metrics import accuracy_score , confusion_matrix , classification_report


accuracy = accuracy_score(original,prediction)
cm = confusion_matrix(original,prediction)
cr = classification_report(original,prediction,labels=[0,1])

#Image Segmentation
brain_df = pd.read_csv("data_mask.csv")
brain_df_mask = brain_df[brain_df['mask'] == 1]


X_train , X_val = train_test_split(brain_df_mask,test_size=0.15)
X_test , X_val = train_test_split(X_val,test_size=0.5)
train_ids = list(X_train.image_path)
train_mask = list(X_train.mask_path)

val_ids = list(X_val.image_path)
val_mask = list(X_val.mask_path)
from utilities import DataGenerator

training_generator = DataGenerator(train_ids,train_mask)
validation_generator = DataGenerator(val_ids,val_mask)

def res_block(X,f):
    X_copy = X
    
    #Main path
    X = Conv2D(f,kernel_size=(1,1),strides=(1, 1),kernel_initializer='he_normal')(X)
    X = BatchNormalization()(X)
    X = Activation('relu')(X)
    
    X = Conv2D(f,kernel_size=(3,3),strides=(1, 1),padding='same',kernel_initializer='he_normal')(X)
    X = BatchNormalization()(X)
    
    #short path 
    X_copy = Conv2D(f,kernel_size=(1,1),strides=(1, 1),kernel_initializer='he_normal')(X_copy)
    X_copy = BatchNormalization()(X_copy)
    
    #Adding
    X = Add()([X,X_copy])
    X = Activation('relu')(X)
    
    return X

def upsample(x,skip):
    x = UpSampling2D(size=(2,2))(x)
    merge = Concatenate()([x,skip])
    
    return merge
X_input = Input((256,256,3))


#stage1
conv1_in = Conv2D(16,3,activation='relu',padding='same',kernel_initializer='he_normal')(X_input)
conv1_in = BatchNormalization()(conv1_in)
conv1_in = Conv2D(16,3,activation='relu',padding='same',kernel_initializer='he_normal')(conv1_in)
conv1_in = BatchNormalization()(conv1_in)
pool1 = MaxPool2D(pool_size=(2,2))(conv1_in)

#stage2

conv2_in = res_block(pool1,32)
pool2 = MaxPool2D(pool_size=(2,2))(conv2_in)

#stage3
conv3_in = res_block(pool2,64)
pool3 = MaxPool2D(pool_size=(2,2))(conv3_in)

#stage4
conv4_in = res_block(pool3,128)
pool4 = MaxPool2D(pool_size=(2,2))(conv4_in)

#stage5
conv5_in = res_block(pool4,256)

#stage6
up1 = upsample(conv5_in,conv4_in)
up1 = res_block(up1,128)

#stage7
up2 = upsample(up1,conv3_in)
up2 = res_block(up2,64)

#stage8
up3 = upsample(up2,conv2_in)
up3 = res_block(up3,32)

#stage9
up4 = upsample(up3,conv1_in)
up4 = res_block(up4,16)

#Output

output = Conv2D(1,(1,1),padding='same',activation='sigmoid')(up4)

#model

model = Model(inputs= X_input , outputs = output)

from utilities import focal_tversky , tversky ,tversky_loss


adam = tf.keras.optimizers.AdamW(learning_rate=0.05,epsilon=0.1)

model.compile(loss=focal_tversky,metrics=[tversky],optimizer=adam)
early_stopping = EarlyStopping(monitor='val_loss',mode='min',patience=20)
check_pointer = ModelCheckpoint(filepath='check_pointer_resunet.keras',verbose=1,save_best_only=True)
history = model.fit(training_generator,validation_data=validation_generator,epochs=1,callbacks=[check_pointer,early_stopping])

from tensorflow.keras.models import model_from_json
img_seg_model = model.to_json()

with open("image_seg.json",'w') as image_json:
    image_json.write(img_seg_model)

with open("image_seg.json",'r') as image_json:
    model_seg = image_json.read()

model_seg = model_from_json(model_seg)

model_seg.load_weights("weights_seg.hdf5")

adam = tf.keras.optimizers.Adam(learning_rate=0.05,epsilon=0.1)
model_seg.compile(loss=focal_tversky,metrics=[tversky],optimizer=adam)

from utilities import prediction

model_seg.predict(test.image_path[0])

image_id, mask, has_mask = prediction(test,model_best,model_seg)
df_pred = pd.DataFrame({'image_path': image_id , 'predection_mask':mask , 'has_mask':has_mask})
df_pred = test.merge(df_pred , on='image_path')


count = 0
fig , axes = plt.subplots(1,5, figsize = (10,5))
for i in range(len(df_pred)):
    if df_pred['has_mask'][i] == 1 and count < 5:
        img = io.imread(df_pred.image_path[i])
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        axes[count][0].title.set_text("Brain MRI")
        axes[count][0].imshow(img)
        
        mask = io.imread(df_pred.mask_path[i])
        axes[count][1].title.set_text("Original Mask")
        axes[count][1].imshow(mask)
        
        pred_mask = np.asarray(df_pred.predection_mask[i])[0].squeeze().round()
        axes[count][2].title.set_text("AI Predicted Mask")
        axes[count][2].imshow(pred_mask)
        
        img[mask == 255] = (255,0,0)
        axes[count][3].title.set_text("MRI with Original Mask (Ground Truth)")
        axes[count][3].imshow(img)
        
        img_ = io.imread(df_pred.image_path[i])
        img_ = cv2.cvtColor(img_,cv2.COLOR_BGR2RGB)
        img_[pred_mask == 1] = (0,255,0)
        axes[count][4].title.set_text("MRI with AI Predicted Mask")
        axes[count][4].imshow(img_)
        count += 1
fig.tight_layout()
        
        


import pandas as pd
brain_df = pd.read_csv("data_mask.csv")
brain_df_mask = brain_df[brain_df['mask'] == 1]
# split the data into train and test data

from sklearn.model_selection import train_test_split

X_train, X_val = train_test_split(brain_df_mask, test_size=0.15)
X_test, X_val = train_test_split(X_val, test_size=0.5)

# create separate list for imageId, classId to pass into the generator

train_ids = list(X_train.image_path)
train_mask = list(X_train.mask_path)

val_ids = list(X_val.image_path)
val_mask= list(X_val.mask_path)

# Utilities file contains the code for custom loss function and custom data generator
from utilities import DataGenerator

# create image generators

training_generator = DataGenerator(train_ids,train_mask)
validation_generator = DataGenerator(val_ids,val_mask)

def resblock(X, f):
  

  # make a copy of input
  X_copy = X

  # main path
  # Read more about he_normal: https://medium.com/@prateekvishnu/xavier-and-he-normal-he-et-al-initialization-8e3d7a087528

  X = Conv2D(f, kernel_size = (1,1) ,strides = (1,1),kernel_initializer ='he_normal')(X)
  X = BatchNormalization()(X)
  X = Activation('relu')(X) 

  X = Conv2D(f, kernel_size = (3,3), strides =(1,1), padding = 'same', kernel_initializer ='he_normal')(X)
  X = BatchNormalization()(X)

  # Short path
  # Read more here: https://towardsdatascience.com/understanding-and-coding-a-resnet-in-keras-446d7ff84d33

  X_copy = Conv2D(f, kernel_size = (1,1), strides =(1,1), kernel_initializer ='he_normal')(X_copy)
  X_copy = BatchNormalization()(X_copy)

  # Adding the output from main path and short path together

  X = Add()([X,X_copy])
  X = Activation('relu')(X)

  return X

# function to upscale and concatenate the values passsed
def upsample_concat(x, skip):
  x = UpSampling2D((2,2))(x)
  merge = Concatenate()([x, skip])

  return merge

# Utilities file contains the code for custom loss function and custom data generator

from utilities import focal_tversky, tversky_loss, tversky

# Compile the model
adam = tf.keras.optimizers.Adam(learning_rate = 0.05, epsilon = 0.1)
model_seg.compile(optimizer = adam, loss = focal_tversky, metrics = [tversky])


input_shape = (256,256,3)

# Input tensor shape
X_input = Input(input_shape)

# Stage 1
conv1_in = Conv2D(16,3,activation= 'relu', padding = 'same', kernel_initializer ='he_normal')(X_input)
conv1_in = BatchNormalization()(conv1_in)
conv1_in = Conv2D(16,3,activation= 'relu', padding = 'same', kernel_initializer ='he_normal')(conv1_in)
conv1_in = BatchNormalization()(conv1_in)
pool_1 = MaxPool2D(pool_size = (2,2))(conv1_in)

# Stage 2
conv2_in = resblock(pool_1, 32)
pool_2 = MaxPool2D(pool_size = (2,2))(conv2_in)

# Stage 3
conv3_in = resblock(pool_2, 64)
pool_3 = MaxPool2D(pool_size = (2,2))(conv3_in)

# Stage 4
conv4_in = resblock(pool_3, 128)
pool_4 = MaxPool2D(pool_size = (2,2))(conv4_in)

# Stage 5 (Bottle Neck)
conv5_in = resblock(pool_4, 256)

# Upscale stage 1
up_1 = upsample_concat(conv5_in, conv4_in)
up_1 = resblock(up_1, 128)

# Upscale stage 2
up_2 = upsample_concat(up_1, conv3_in)
up_2 = resblock(up_2, 64)

# Upscale stage 3
up_3 = upsample_concat(up_2, conv2_in)
up_3 = resblock(up_3, 32)

# Upscale stage 4
up_4 = upsample_concat(up_3, conv1_in)
up_4 = resblock(up_4, 16)

# Final Output
output = Conv2D(1, (1,1), padding = "same", activation = "sigmoid")(up_4)

model_seg = Model(inputs = X_input, outputs = output )

# use early stopping to exit training if validation loss is not decreasing even after certain epochs (patience)
earlystopping = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=20)

# save the best model with lower validation loss
checkpointer = ModelCheckpoint(filepath="ResUNet-weights.keras", verbose=1, save_best_only=True)

history = model_seg.fit(training_generator, epochs = 1, validation_data = validation_generator, callbacks = [checkpointer, earlystopping])