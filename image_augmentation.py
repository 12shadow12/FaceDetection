import albumentations as A
import cv2, os
import json
import numpy as np

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

# augmentation pipeline
for partition in ['train', 'test', 'val']:
    for image in os.listdir(os.path.join('data', partition, 'images')):
        img = cv2.imread(os.path.join('data', partition, 'images', image))

        coords = [0,0,0.00001,0.00001]
        label_path = os.path.join('data', partition, 'labels', f'{image.split(".")[0]}.json')
        if os.path.exists(label_path):
            with open(label_path, 'r') as file:
                label = json.load(file)
                
            # change the image coordinates into 1D array
            coords[0] = label['shapes'][0]['points'][0][0]
            coords[1] = label['shapes'][0]['points'][0][1]
            coords[2] = label['shapes'][0]['points'][1][0]
            coords[3] = label['shapes'][0]['points'][1][1]

            # transform pascal_voc into albumentations format
            coords = list(np.divide(coords, [640, 480, 640, 480]))
        
        # Generate 120 images per image.
        try:
            for x in range(120):
                augmented = augmentor(image = img, bboxes = [coords], class_labels = ['face'])
                cv2.imwrite(os.path.join('augmented_data', partition, 'images', f'{image.split(".")[0]}.{x}.jpg'), augmented['image'])

                annotation = {}
                annotation['image'] = image

                if os.path.exists(label_path):
                    if len(augmented['bboxes']) == 0:
                        annotation['bbox'] = [0, 0, 0, 0]
                        annotation['class'] = 0
                    else:
                        annotation['bbox'] = augmented['bboxes'][0]
                        annotation['class'] = 1
                else:
                    annotation['bbox'] = [0, 0, 0, 0]
                    annotation['class'] = 0

                with open(os.path.join('augmented_data', partition, 'labels', f'{image.split(".")[0]}.{x}.json'), 'w') as json_file:
                    json.dump(annotation, json_file)

        except Exception as e:
            print(e)