import os
import tensorflow as tf
import json

# Move the images and labels into train and test and val
for folder in ['train', 'test', 'val']:
    for file in os.listdir(os.path.join('data', folder, 'images')):
        file_name = file.split('.')[0] +'.json'
        existing_filepath = os.path.join('data', 'labels', file_name)

        if os.path.exists(existing_filepath):
            new_file_path = os.path.join('data', folder, 'labels', file_name)
            os.replace(existing_filepath, new_file_path)


# Function to load and preprocess an image
def load_image(file_path):
    byte_img = tf.io.read_file(file_path)
    img = tf.io.decode_jpeg(byte_img)
    img = tf.image.resize(img, (120, 120))
    img = img / 255.0  # Normalizing the pixel values to [0, 1]
    return img

# Function to load labels
def load_labels(label_path):
    with open(label_path.numpy(), 'r', encoding = "utf-8") as file:
        label = json.load(file)
    return [label['class']], label['bbox']

