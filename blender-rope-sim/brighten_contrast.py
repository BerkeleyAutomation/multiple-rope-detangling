import cv2
import os
import argparse
import json
import numpy as np

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dir', type=str, default='./real_images/no_mask/two_hairties_resized')
    args = parser.parse_args()
    output_dir = args.dir + '_editted'
    if not os.path.exists(output_dir):
    	os.mkdir(output_dir)

    alpha = 1.65 # Contrast control (1.0-3.0)
    beta = 0 # Brightness control (0-100)
    for f in os.listdir(args.dir):
        img = cv2.imread('%s/%s'%(args.dir, f)).copy()
        print(f)
        adjusted = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
        filename = output_dir + '/' + f
        cv2.imwrite('{}'.format(filename), adjusted)
