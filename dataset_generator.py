import os
import time
import uuid
import cv2

# getting the image path data/images
image_path = os.path.join('data', 'images')
number_of_images = 90

# open the camera and collect images for training data
camera = cv2.VideoCapture(0)
for image_num in range(number_of_images):
    print("Collecting images: {}".format(image_num))
    ret, frame = camera.read()
    unique_image_name = os.path.join(image_path, f'{str(uuid.uuid1())}.jpg')
    cv2.imwrite(unique_image_name, frame)
    cv2.imshow("Captured Frames", frame)
    time.sleep(0.5)

    
    # filter and extract the least significant byte
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

#closes camera capture and deallocate memory.   
camera.release()
cv2.destroyAllWindows()