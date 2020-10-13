import cv2
import os
import sys
import argparse
import numpy as np
from shutil import copyfile
import json

def darken_image(img, filename):
	M = np.ones(img.shape, dtype="uint8") * 115
	subtracted=cv2.subtract(img,M)
	cv2.imwrite('./dr_data/dark_val2017/{}'.format(filename), subtracted)

def reorder_directory(dir):
	num = 78
	for filename in sorted(os.listdir(dir)):
		file = os.path.join(dir, filename)
		os.rename(dir + filename, dir + '%05d.jpg'%num)
		num += 1

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('-d', '--dir', type=str, default='./dr_data/val2017/')
	args = parser.parse_args()
	if os.path.exists('./dr_data/dark_val2017/'):
		os.system('rm -r ./dr_data/dark_val2017/')
	os.mkdir('./dr_data/dark_val2017/')
	img_dir = args.dir
	for i in range(len(os.listdir(img_dir))):
		print(os.path.join(img_dir, '%06d.jpg'%i))
		img = cv2.imread(os.path.join(img_dir, '%06d.jpg'%i))
		print(img)
		filename = '%06d.jpg'%i
		darken_image(img, filename)
	reorder_directory('./dr_data/dark_val2017/')