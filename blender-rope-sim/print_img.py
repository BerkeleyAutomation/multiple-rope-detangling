import cv2
import os
import argparse
import json
import numpy as np

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dir', type=str, default='./preds_4c')
    args = parser.parse_args()
    output_dir = "./preds_4c_check"
    if not os.path.exists(output_dir):
    	os.mkdir(output_dir)

    for f in os.listdir(args.dir):
        img = np.load('%s/%s'%(args.dir, f)).copy()
        img = img[:,:,0:3]
        parts = f.split('.')
        num = int(parts[0])
        cv2.imwrite(os.path.join("preds_4c_check/%05d.png"%num), img)