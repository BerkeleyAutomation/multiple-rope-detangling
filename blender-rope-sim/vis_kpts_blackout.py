import cv2
import numpy as np
import argparse
from xml.etree import ElementTree
import os
import math
import json
import colorsys

def show_kpts(idx, version, kpt_dir):
    image_filename = "{0:05d}_{1:01d}_blacked_out.png".format(idx, version)
    img = cv2.imread('render_kpts_output/blacked_out/{}'.format(image_filename))
    vis = img.copy()
    kpts = np.load('%s/%05d_%01d.npy'%(kpt_dir, idx, version))
    kpts = np.reshape(kpts, (4,2))
    for i, (u,v) in enumerate(kpts):    
        (r, g, b) = colorsys.hsv_to_rgb(float(i)/kpts.shape[0], 1.0, 1.0)
        R, G, B = int(255 * r), int(255 * g), int(255 * b)
        cv2.circle(vis,(u,v),4,(R,G,B), -1)
    annotated_filename = "{0:05d}_{1:01d}_annotated.png".format(idx, version)
    print("Annotating: %d, %d"% (idx, version))
    cv2.imwrite('./annotated/{}'.format(annotated_filename), vis)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dir', type=str, default='render_kpts_output/keypoints')
    args = parser.parse_args()
    print(args)
    if not os.path.exists("./annotated"):
        os.makedirs('./annotated')
    else:
        os.system("rm -rf ./annotated")
        os.makedirs("./annotated")
    for i in range(len(os.listdir(args.dir))):
        for j in range(3):
            show_kpts(i, j, args.dir)
