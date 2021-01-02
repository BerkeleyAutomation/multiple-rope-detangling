import cv2
import os
import argparse
import json
import numpy as np

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dir', type=str, default='./test_results/two_hairties_ep_5000/preds_4c')
    args = parser.parse_args()

    for f in os.listdir(args.dir):
        img = np.load('%s/%s'%(args.dir, f)).copy()
        print(img.shape)