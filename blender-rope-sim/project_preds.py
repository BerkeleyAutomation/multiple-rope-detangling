import numpy as np 
import cv2
import os
   
def project_pred(img, pred, plot=False):
     # Convert BGR to HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # define range of black color in HSV
    lower_val = np.array([0,0,0])
    upper_val = np.array([179,255,127])

    # Threshold the HSV image to get only black colors
    mask = cv2.inRange(hsv, lower_val, upper_val)

    # invert mask to get mask of rope
    mask_inv = cv2.bitwise_not(mask)

    TARGET = (pred[0], pred[1])

    nonzero = cv2.findNonZero(mask_inv)
    distances = np.sqrt((nonzero[:,:,0] - TARGET[0])**2 + (nonzero[:,:,1] - TARGET[1])**2)
    nearest_index = np.argmin(distances)
    closest = nonzero[nearest_index][0]
    point = (closest[0], closest[1])
    direction_of_shift = [TARGET[0] - point[0], TARGET[1] - point[1]]
    if direction_of_shift == [0, 0]:
        return TARGET
    norm = np.linalg.norm(direction_of_shift)
    norm_vec = direction_of_shift/norm
    point -= 5*norm_vec
    point = (int(point[0]), int(point[1]))

    if plot:
        cv2.circle(img, TARGET, 5, (100,0,255), -1)
        cv2.circle(img, point, 5, (255,255,255), -1)
        cv2.imshow("img", img)
        cv2.waitKey(0)

    return point

if __name__ == '__main__':
    image_dir = './real_images/no_mask/lab_two_rope_crop'
    crop_size = (560,420)
    network_input_size = (640,480)
    for f in os.listdir(image_dir):
        if f != '.DS_Store':
            image_path = os.path.join(image_dir, f)
            img = cv2.imread(image_path)
            pred = (450, 235)
            project_pred(img, pred)