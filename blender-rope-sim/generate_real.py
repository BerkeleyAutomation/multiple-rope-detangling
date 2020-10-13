import cv2
import os
import imgaug as ia
from imgaug import augmenters as iaa
import numpy as np
aug = iaa.CoarseDropout((0.0, 0.3), size_percent=(0.03, 0.04))
def segment_rope(img, plot=False):
    # Dropout random pixels
    img = aug.augment_image(img)
    if plot:
        cv2.imshow("img", img)
        cv2.waitKey(0)
    # Convert to grayscale and segment rope
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray,70,255,cv2.THRESH_BINARY)
    # Get RGB segmask
    img = cv2.bitwise_and(img, img, mask=mask)

    #np.where to find all the indices greater than 0
    if plot:
        cv2.imshow("fg", img)
        cv2.waitKey(0)
    return img
if __name__ == '__main__':
    folder = 'overhead_hairtie_resized'
    output_folder = folder + '_masks'
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)
    for f in os.listdir(folder):
        if f != '.DS_Store':
            img = cv2.imread(os.path.join(folder, f))
            m = segment_rope(img, plot=True)
            cv2.imwrite(os.path.join(output_folder, f), m)