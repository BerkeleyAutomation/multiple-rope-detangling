import cv2
import os
import argparse
import json
import numpy as np

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dir', type=str, default='./real_images/aug_get_right_ep')
    args = parser.parse_args()

    for f in os.listdir(args.dir):
        img = cv2.imread('%s/%s'%(args.dir, f)).copy()
        print(img.shape)