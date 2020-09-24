import cv2
import os
import sys
import argparse
import numpy as np
from shutil import copyfile

def extract_red_rope(mask, image, filename):
	mask_rope = mask.astype(np.uint8)
	image = image.astype(np.uint8)

	#blurring and smoothin
	hsv = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)

	#lower red
	lower_red = np.array([0,50,50])
	upper_red = np.array([10,255,255])

	#upper red
	lower_red2 = np.array([170,50,50])
	upper_red2 = np.array([180,255,255])

	mask = cv2.inRange(hsv, lower_red, upper_red)
	res = cv2.bitwise_and(image, mask_rope, mask= mask)


	mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
	res2 = cv2.bitwise_and(image, mask_rope, mask= mask2)

	img3 = res+res2
	img4 = cv2.add(res,res2)
	img5 = cv2.addWeighted(res,0.5,res2,0.5,0)


	kernel = np.ones((15,15),np.float32)/225
	smoothed = cv2.filter2D(res,-1,kernel)
	smoothed2 = cv2.filter2D(img3,-1,kernel)

	cv2.imwrite('./render_kpts_output/extracted_images/{}'.format(filename), img4)

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('-d', '--dir', type=str, default='./render_kpts_output/images')
	parser.add_argument('-m', '--masks', type=str, default="./render_kpts_output/image_masks")
	args = parser.parse_args()
	if os.path.exists('./render_kpts_output/extracted_images'):
		os.system('rm -r ./render_kpts_output/extracted_images')
	os.mkdir('./render_kpts_output/extracted_images')
	img_dir = args.dir
	mask_dir = args.masks
	for i in range(len(os.listdir(img_dir))):
		print(os.path.join(img_dir, '%05d.jpg'%i))
		mask = cv2.imread(os.path.join(mask_dir, '%05d.jpg'%i))
		img = cv2.imread(os.path.join(img_dir, '%05d.jpg'%i))
		filename = '%05d.jpg'%i
		extract_red_rope(mask, img, filename)
