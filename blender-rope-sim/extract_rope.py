import cv2
import os
import sys
import argparse
import numpy as np
from shutil import copyfile
import pandas as pd 
import matplotlib.pyplot as plt 
from sklearn.cluster import KMeans

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


def sort_colors(image):
	img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	img = img .reshape((img.shape[1]*img.shape[0],3))
	kmeans = KMeans(n_clusters=3)
	s = kmeans.fit(img)
	labels = kmeans.labels_
	labels = list(labels)
	centroid = kmeans.cluster_centers_
	print(centroid)

	percent = []
	for i in range(len(centroid)):
		j = labels.count(i)
		j = j/len(labels)
		percent.append(j)
	print(percent)
	colors = np.argsort(percent)
	#print(colors)

	#plt.pie(percent, colors=np.array(centroid/255), labels=np.arange(len(centroid)))
	#plt.show()


def extract_rope(mask, image, filename):
	img = cv2.bitwise_and(image, mask, mask=None)
	cv2.imwrite('./render_kpts_output/extracted_images/{}'.format(filename), img) 


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
	for i in range(len(os.listdir(img_dir))//3):
		for j in range(3):
			print(os.path.join(img_dir, '%05d_%01d.jpg'%(i,j)))
			mask = cv2.imread(os.path.join(mask_dir, '%05d.jpg'%i))
			img = cv2.imread(os.path.join(img_dir, '%05d_%01d.jpg'%(i,j)))
			filename = '%05d_%01d.jpg'%(i,j)
			extract_rope(mask, img, filename)
