import numpy as np
import cv2
import os

def stage_empty(img):
     # Convert BGR to HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # define range of black color in HSV
    lower_val = np.array([0,0,0])
    upper_val = np.array([179,255,127])

    # Threshold the HSV image to get only black colors
    mask = cv2.inRange(hsv, lower_val, upper_val)

    # invert mask to get black symbols on white background
    mask_inv = cv2.bitwise_not(mask)

    if cv2.countNonZero(mask_inv) == 0:
        return True
    return False

if __name__ == '__main__':
    image_dir = './test1_resized'
    crop_size = (560,420)
    network_input_size = (640,480)
    for f in os.listdir(image_dir):
        if f != '.DS_Store':
            image_path = os.path.join(image_dir, f)
            img = cv2.imread(image_path)
            print(stage_empty(img))