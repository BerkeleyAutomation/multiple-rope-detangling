import cv2
import numpy as np
import argparse
from xml.etree import ElementTree
import os
import math
import json
import colorsys

def show_kpts(idx, kpts):
    image_filename = "{0:05d}.png".format(idx)
    print(image_filename)
    img = cv2.imread('images/{}'.format(image_filename))
    vis = img.copy()
    for i, (u,v) in enumerate(kpts):    
        (r, g, b) = colorsys.hsv_to_rgb(float(i)/kpts.shape[0], 1.0, 1.0)
        R, G, B = int(255 * r), int(255 * g), int(255 * b)
        cv2.circle(vis,(u,v),4,(R,G,B), -1)
    annotated_filename = "{0:05d}_annotated.png".format(idx)
    print("Annotating: %d"%idx)
    cv2.imwrite('./annotated/{}'.format(annotated_filename), vis)

def get_reid_moves(idx, kp_dir):
	right_ep = None
	left_ep = None
	image_filename = "{0:05d}.png".format(idx)
	img = cv2.imread('images/{}'.format(image_filename))
	kpts = np.load('%s/%05d.npy'%(kp_dir, idx))
	kpts = np.reshape(kpts, (4,2))
	max_x = 0
	for i, (u,v) in enumerate(kpts):
		if u > max_x:
			max_x = u
			right_ep = [u,v]
	color1_red = check_red(right_ep, img)
	min_x = 640
	for i, (u,v) in enumerate(kpts):
		color2 = img[v, u]
		color2_red = check_red([u,v], img)
		if u < min_x and color2_red != color1_red:
			min_x = u
			left_ep = [u,v]
	print(right_ep)
	print(left_ep)
	right_ep.extend(left_ep)
	return np.reshape(right_ep, (2,2)) #order is right endpoint first, then left endpoint

def check_red(pixel, img):
	hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
	#lower red
	lower_red = np.array([0,50,50])
	upper_red = np.array([10,255,255])
	#upper red
	lower_red2 = np.array([170,50,50])
	upper_red2 = np.array([180,255,255])

	mask = cv2.inRange(hsv, lower_red, upper_red)
	mask2 = cv2.inRange(hsv, lower_red2, upper_red2)

	mask_val = mask[pixel[1], pixel[0]]
	mask2_val = mask2[pixel[1], pixel[0]]
	if mask_val != 0:
		return True
	elif mask2_val != 0:
		return True
	else:
		return False

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dir', type=str, default='keypoints')
    args = parser.parse_args()
    print(args)
    if not os.path.exists("./annotated"):
        os.makedirs('./annotated')
    else:
        os.system("rm -rf ./annotated")
        os.makedirs("./annotated")
    for i in range(len(os.listdir(args.dir))):
    	pull_points = get_reid_moves(i, args.dir)
    	show_kpts(i, pull_points)



