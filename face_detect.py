import numpy as np
import cv2
import time
import os
import uuid
import matplotlib.pyplot as plt
import tensorflow as tf
import json
import albumentations as A
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, Dense, GlobalMaxPooling2D, MaxPooling2D, Flatten, Add, Dropout
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import load_model

# getting the image path data/images
# image_path = os.path.join('data', 'images')
# number_of_images = 30

# open the camera and collect images for training data
# camera = cv2.VideoCapture(0)
# for image_num in range(number_of_images):
#     print("Collecting images {}".format(image_num))
#     ret, frame = camera.read()
#     unique_image_name = os.path.join(image_path, f'{str(uuid.uuid1())}.jpg')
#     cv2.imwrite(unique_image_name, frame)
#     cv2.imshow("Capture Frames", frame)
#     time.sleep(0.5)

    # filter and extract the least significant byte
    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     break

# closes camera capture and deallocate memory.   
# camera.release()
# cv2.destroyAllWindows()

# Avoid out of memory errors by setting GPU memory comsumption growth
# gpus = tf.config.experimental.list_physical_devices('GPU')
# for gpu in gpus:
#     tf.config.experimental.set_memory_growth(gpu, True)

# print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# from tensorflow.python.client import device_lib

# local_device_protos = device_lib.list_local_devices()
# for device in local_device_protos:
#     print(device)

# load image into TF data pipline
#images = tf.data.Dataset.list_files('data\\images\\*.jpg', shuffle=False)


# Map the images using load_image function
# and Extract the next element
# images = images.map(load_image)
# image_data = images.as_numpy_iterator().next()
# print(image_data)
#print(type(images))

# Batching, dividing data for model training
# batch_size = 4
# image_generator = images.batch(batch_size).as_numpy_iterator()

# Get the batch of images
# plot_images = image_generator.next()

# use this outerloop to loop over multiple batches
#for batch_idx, plot_images in enumerate(image_generator):

# Create subplots for displaying the images in the batch
# fig, arr = plt.subplots(ncols = 4, figsize = (20,20))

# loop over the images in the batch and display each one
# for idx, images in enumerate(plot_images):
#     arr[idx].imshow(images)

# Visualize images with matplotlib
# plt.show()

# Paritition images and labels into train and test and val
# Move the labels into train , test and val
for folder in ['train', 'test', 'val']:
    for file in os.listdir(os.path.join('data', folder, 'images')):
        file_name = file.split('.')[0] +'.json'
        existing_filepath = os.path.join('data', 'labels', file_name)

        if os.path.exists(existing_filepath):
            new_file_path = os.path.join('data', folder, 'labels', file_name)
            os.replace(existing_filepath, new_file_path)


# load test image and annotation with opencv and json
img = cv2.imread(os.path.join('data', 'train', 'images', '5a97e9c9-6ef1-11ee-bb0d-4c3488933b4f.jpg'))
with open(os.path.join('data', 'train', 'labels', '5a97e9c9-6ef1-11ee-bb0d-4c3488933b4f.json'), 'r') as file:
    label = json.load(file)
print(label['shapes'][0]['points'])

# change the image coordinates into 1D array
coords = [0,0,0,0]
coords[0] = label['shapes'][0]['points'][0][0]
coords[1] = label['shapes'][0]['points'][0][1]
coords[2] = label['shapes'][0]['points'][1][0]
coords[3] = label['shapes'][0]['points'][1][1]
print("Pascal_voc: ", coords)

# transform pascal_voc into albumentations format
coords = list(np.divide(coords, [640, 480, 640, 480]))
print(" Albumentation Coordinates are: ", coords)

# grab a random image and check its dimensions
img = cv2.imread(os.path.join('data', 'train', 'images', '5a97e9c9-6ef1-11ee-bb0d-4c3488933b4f.jpg'))
print("Image dimensions: ", img.shape)

# use albumentations to transform labels
# 6 albumentations
augmentor = A.Compose([A.RandomCrop(width = 450, height = 450),
    A.HorizontalFlip(p = 0.5),
    A.RandomBrightnessContrast(p = 0.2),
    A.VerticalFlip(p = 0.5),
    A.RandomGamma(p = 0.2),
    A.RGBShift(p = 0.2)],
    bbox_params = A.BboxParams(format = 'albumentations',
    label_fields = ['class_labels']))

# apply the augmentation using the augmentator
augmented = augmentor(image = img, bboxes = [coords], class_labels = ['face'])
print(augmented.keys())
print("Augmented coordinates are: ", augmented['bboxes'])

# drawing rectangle, the color is in BGR format
# untransform the normalized values and scale it by 450
# last parameter is the thickness of the rectangle border
cv2.rectangle(augmented['image'], tuple(np.multiply(augmented['bboxes'][0][:2], [450, 450]).astype(int)),
             tuple(np.multiply(augmented['bboxes'][0][2:], [450, 450]).astype(int)), (255, 0, 0), 2)

# display the image using imshow()
#plt.imshow(augmented['image'])

# show image in a graphical window
#plt.show()

# augmentation pipeline
for partition in ['train', 'test', 'val']:
    for image in os.listdir(os.path.join('data', partition, 'images')):
        img = cv2.imread(os.path.join('data', partition, 'images', image))

        coords = [0,0,0.00001,0.00001]
        label_path = os.path.join('data', partition, 'labels', f'{image.split(".")[0]}.json')
        if os.path.exists(label_path):
            with open(label_path, 'r') as file:
                label = json.load(file)
            coords[0] = label['shapes'][0]['points'][0][0]
            coords[1] = label['shapes'][0]['points'][0][1]
            coords[2] = label['shapes'][0]['points'][1][0]
            coords[3] = label['shapes'][0]['points'][1][1]
            coords = list(np.divide(coords, [640, 480, 640, 480]))
            
        # try:
        #     for x in range(120):
        #         augmented = augmentor(image = img, bboxes = [coords], class_labels = ['face'])
        #         cv2.imwrite(os.path.join('augmented_data', partition, 'images', f'{image.split(".")[0]}.{x}.jpg'), augmented['image'])

        #         annotation = {}
        #         annotation['image'] = image

        #         if os.path.exists(label_path):
        #             if len(augmented['bboxes']) == 0:
        #                 annotation['bbox'] = [0, 0, 0, 0]
        #                 annotation['class'] = 0
        #             else:
        #                 annotation['bbox'] = augmented['bboxes'][0]
        #                 annotation['class'] = 1
        #         else:
        #             annotation['bbox'] = [0, 0, 0, 0]
        #             annotation['class'] = 0

        #         with open(os.path.join('augmented_data', partition, 'labels', f'{image.split(".")[0]}.{x}.json'), 'w') as json_file:
        #             json.dump(annotation, json_file)

        # except Exception as e:
        #     print(e)

# Function to load and preprocess an image
def load_image(file_path):
    byte_img = tf.io.read_file(file_path)
    img = tf.io.decode_jpeg(byte_img)
    img = tf.image.resize(img, (120, 120))
    img = img / 255.0  # Normalizing the pixel values to [0, 1]
    return img

# load image and create tensorflow dataset
train_images = tf.data.Dataset.list_files('augmented_data\\train\\images\\*.jpg', shuffle=False)
train_images = train_images.map(load_image)

test_images = tf.data.Dataset.list_files('augmented_data\\test\\images\\*.jpg', shuffle=False)
test_images = test_images.map(load_image)

val_images = tf.data.Dataset.list_files('augmented_data\\val\\images\\*.jpg', shuffle=False)
val_images = val_images.map(load_image)

#print(train_images.as_numpy_iterator().next())

# Function to load labels
def load_labels(label_path):
    with open(label_path.numpy(), 'r', encoding = "utf-8") as file:
        label = json.load(file)
    return [label['class']], label['bbox']

train_labels = tf.data.Dataset.list_files('augmented_data\\train\\labels\\*.json', shuffle = False)
train_labels = train_labels.map(lambda x: tf.py_function(load_labels, [x], [tf.uint8, tf.float16]))

test_labels = tf.data.Dataset.list_files('augmented_data\\test\\labels\\*.json', shuffle = False)
test_labels = test_labels.map(lambda x: tf.py_function(load_labels, [x], [tf.uint8, tf.float16]))

val_labels = tf.data.Dataset.list_files('augmented_data\\val\\labels\\*.json', shuffle = False)
val_labels = val_labels.map(lambda x: tf.py_function(load_labels, [x], [tf.uint8, tf.float16]))

print(train_labels.as_numpy_iterator().next())
print(len(train_images), len(train_labels), len(test_images), len(test_labels), len(val_images), len(val_labels))

train = tf.data.Dataset.zip((train_images, train_labels))
train = train.shuffle(8000)
train = train.batch(8)
train = train.prefetch(4)
test = tf.data.Dataset.zip((test_images, test_labels))
test = test.shuffle(2000)
test = test.batch(8)
test = test.prefetch(4)
val = tf.data.Dataset.zip((val_images, val_labels))
val = val.shuffle(1800)
val = val.batch(8)
val = val.prefetch(4)

# print(train.as_numpy_iterator().next()[1])

# data_samples = train.as_numpy_iterator()
# res = data_samples.next()
# fig, ax = plt.subplots(ncols=4, figsize=(20,20))
# for idx in range(4): 
#     sample_image = res[0][idx]
#     sample_coords = res[1][1][idx]
    
#     cv2.rectangle(sample_image, 
#                   tuple(np.multiply(sample_coords[:2], [120,120]).astype(int)),
#                   tuple(np.multiply(sample_coords[2:], [120,120]).astype(int)), 
#                         (255,0,0), 2)

#     ax[idx].imshow(sample_image)
# plt.show()

vgg = VGG16(include_top=False)
vgg.summary()

# Building VGG neural network
def build_model(): 
    input_layer = Input(shape=(120,120,3))
    
    # Feature extraction using VGG16
    vgg = VGG16(include_top=False)(input_layer)

    # Classification branch 
    class_f = GlobalMaxPooling2D()(vgg)
    class_dense = Dense(2048, activation='relu')(class_f)
    class_output = Dense(1, activation='sigmoid', name='classification')(class_dense)
    
    # Bounding box branch
    regress_f = GlobalMaxPooling2D()(vgg)
    regress_dense = Dense(2048, activation='relu')(regress_f)
    regress_output = Dense(4, activation='sigmoid', name='regression')(regress_dense)
    
    facetracker = Model(inputs=input_layer, outputs=[class_output, regress_output])
    return facetracker

facetracker = build_model()
facetracker.summary()
X, y = train.as_numpy_iterator().next()
X.shape
classes, coords = facetracker.predict(X)
print(classes, coords)

batches_per_epoch = len(train)
#lr_decay = (1./0.75 - 1) / batches_per_epoch
initial_learning_rate = 0.0001

# Define a learning rate schedule with ExponentialDecay
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
initial_learning_rate, decay_steps = batches_per_epoch, decay_rate = 0.75, staircase = False)

opt = tf.keras.optimizers.Adam(learning_rate = lr_schedule)

def localization_loss(y_true, yhat):            
    delta_coord = tf.reduce_sum(tf.square(y_true[:,:2] - yhat[:,:2]))
                  
    h_true = y_true[:,3] - y_true[:,1] 
    w_true = y_true[:,2] - y_true[:,0] 

    h_pred = yhat[:,3] - yhat[:,1] 
    w_pred = yhat[:,2] - yhat[:,0] 
    
    delta_size = tf.reduce_sum(tf.square(w_true - w_pred) + tf.square(h_true-h_pred))
    
    return delta_coord + delta_size


classloss = tf.keras.losses.BinaryCrossentropy()
regressloss = localization_loss

class FaceTracker(Model): 
    def __init__(self, eyetracker,  **kwargs): 
        super().__init__(**kwargs)
        self.model = eyetracker

    def compile(self, opt, classloss, localizationloss, **kwargs):
        super().compile(**kwargs)
        self.closs = classloss
        self.lloss = localizationloss
        self.opt = opt
    
    def train_step(self, batch, **kwargs): 
        
        X, y = batch
        
        with tf.GradientTape() as tape: 
            classes, coords = self.model(X, training=True)
            
            batch_classloss = self.closs(y[0], classes)
            batch_localizationloss = self.lloss(tf.cast(y[1], tf.float32), coords)
            
            total_loss = batch_localizationloss + 0.5 * batch_classloss
            
            grad = tape.gradient(total_loss, self.model.trainable_variables)
        
        opt.apply_gradients(zip(grad, self.model.trainable_variables))
        
        return {"total_loss":total_loss, "class_loss":batch_classloss, "regress_loss":batch_localizationloss}
    
    def test_step(self, batch, **kwargs): 
        X, y = batch
        
        classes, coords = self.model(X, training=False)
        
        batch_classloss = self.closs(y[0], classes)
        batch_localizationloss = self.lloss(tf.cast(y[1], tf.float32), coords)
        total_loss = batch_localizationloss+0.5*batch_classloss
        
        return {"total_loss":total_loss, "class_loss":batch_classloss, "regress_loss":batch_localizationloss}
        
    def call(self, X, **kwargs): 
        return self.model(X, **kwargs)
        
model = FaceTracker(facetracker)
model.compile(opt, classloss, regressloss)

logdir='logs'
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)
# hist = model.fit(train, epochs=10, validation_data=val, callbacks=[tensorboard_callback])

# fig, ax = plt.subplots(ncols=3, figsize=(20,5))

# ax[0].plot(hist.history['total_loss'], color='teal', label='loss')
# ax[0].plot(hist.history['val_total_loss'], color='orange', label='val loss')
# ax[0].title.set_text('Total Loss')
# ax[0].legend()

# ax[1].plot(hist.history['class_loss'], color='teal', label='class loss')
# ax[1].plot(hist.history['val_class_loss'], color='orange', label='val class loss')
# ax[1].title.set_text('Classification Loss')
# ax[1].legend()

# ax[2].plot(hist.history['regress_loss'], color='teal', label='regress loss')
# ax[2].plot(hist.history['val_regress_loss'], color='orange', label='val regress loss')
# ax[2].title.set_text('Regression Loss')
# ax[2].legend()

# plt.show()

# facetracker.save('cnn_face_detection_model.keras')
facetracker = load_model('cnn_face_detection_model.keras')

facetracker.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss={
        'classification': tf.keras.losses.BinaryCrossentropy(),
        'regression': localization_loss
    },
    metrics={
        'classification': [
            tf.keras.metrics.BinaryAccuracy(),
            tf.keras.metrics.FalseNegatives()
        ],
        'regression': localization_loss
    }
)

# Model Accuracy
test_data = test.as_numpy_iterator()
true_labels = []
predicted_labels = []

for batch in test_data:
    X_test, y_test = batch
    predictions = facetracker.predict(X_test)
    true_labels.extend(y_test[0])  #  y_test[0] contains the true class labels
    predicted_labels.extend(predictions[0])  #  predictions[0] contains the predicted class labels

# Convert the lists to numpy arrays for easier comparison
true_labels = np.array(true_labels)
predicted_labels = np.array(predicted_labels)

# Convert predicted probabilities to binary predictions based on a threshold
threshold = 0.5
binary_predictions = (predicted_labels > threshold).astype(int)

# Calculate accuracy
accuracy = np.mean(true_labels == binary_predictions)
print(f"Model Accuracy: {accuracy * 100:.2f}%")


# making predictions on test set
# test_data = test.as_numpy_iterator()
# test_sample = test_data.next()
# yhat = facetracker.predict(test_sample[0])
# fig, ax = plt.subplots(ncols=4, figsize=(20,20))
# for idx in range(4): 
#     sample_image = test_sample[0][idx]
#     sample_coords = yhat[1][idx]
    
#     if yhat[0][idx] > 0.9:
#         cv2.rectangle(sample_image, 
#                       tuple(np.multiply(sample_coords[:2], [120,120]).astype(int)),
#                       tuple(np.multiply(sample_coords[2:], [120,120]).astype(int)), 
#                             (255,0,0), 2)
    
#     ax[idx].imshow(sample_image)
# plt.show()

# cap = cv2.VideoCapture(0)
# while cap.isOpened():
#     _ , frame = cap.read()
#     frame = frame[50:500, 50:500,:]
    
#     rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     resized = tf.image.resize(rgb, (120,120))
    
#     yhat = facetracker.predict(np.expand_dims(resized/255,0))
#     sample_coords = yhat[1][0]
    
#     if yhat[0] > 0.5: 
#         # Controls the main rectangle
#         cv2.rectangle(frame, 
#                       tuple(np.multiply(sample_coords[:2], [450,450]).astype(int)),
#                       tuple(np.multiply(sample_coords[2:], [450,450]).astype(int)), 
#                             (255,0,0), 2)
#         # Controls the label rectangle
#         cv2.rectangle(frame, 
#                       tuple(np.add(np.multiply(sample_coords[:2], [450,450]).astype(int), 
#                                     [0,-30])),
#                       tuple(np.add(np.multiply(sample_coords[:2], [450,450]).astype(int),
#                                     [80,0])), 
#                             (255,0,0), -1)
        
#         # Controls the text rendered
#         cv2.putText(frame, 'face', tuple(np.add(np.multiply(sample_coords[:2], [450,450]).astype(int),
#                                                [0,-5])),
#                     cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
    
#     cv2.imshow('EyeTrack', frame)
    
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
# cap.release()
# cv2.destroyAllWindows()

