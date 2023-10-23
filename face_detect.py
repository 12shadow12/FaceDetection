import numpy as np
import cv2
import time
import os
import uuid
import matplotlib.pyplot as plt
import tensorflow as tf
import json
import albumentations as A

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

# gpu_list = tf.config.list_physical_devices('GPU')
# print("Available GPUs:", gpu_list)

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
plt.imshow(augmented['image'])

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

# pretrained file for face
# cascade_path = pathlib.Path(cv2.__file__).parent.absolute() / "data/haarcascade_frontalface_default.xml"
# print(cascade_path)
# clf = cv2.CascadeClassifier(str(cascade_path))

#pretrained file for eyes
# cle = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")

# camera = cv2.VideoCapture(0)

# roi_gray = None
# while True:
#     # reading image in grayscale mode
#     _, frame = camera.read()
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     faces = clf.detectMultiScale(
#         gray,
#         scaleFactor=1.1,
#         minNeighbors=5,
#         minSize=(30, 30),
#         flags=cv2.CASCADE_SCALE_IMAGE
#     )
    # draw rectangle in a face
    # for (x, y, width, height) in faces:
    #     cv2.rectangle(frame, (x, y), (x+width, y+height), (255, 255, 0), 2)
    #     roi_gray = gray[y:y+height, x:x+width]
    #     roi_color = frame[y:y+height, x:x+width]

    
    # detect the eyes of different size 
    # eyes = cle.detectMultiScale(roi_gray)

    # # draw rectangle in eyes
    # for (eyes_x, eyes_y, eyes_width, eyes_height) in eyes:
    #     cv2.rectangle(roi_color, (eyes_x, eyes_y), (eyes_x + eyes_width, eyes_y + eyes_height), (0, 127, 255), 2)

    # cv2.imshow("Faces", frame)
    # Quits the program if user presses q on the keyboard
    # if cv2.waitKey(1) == ord("q"):
    #     break

# closes camera capture and deallocate memory.
# camera.release()
# cv2.destroyAllWindows()