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

	percent = []
	for i in range(len(centroid)):
		j = labels.count(i)
		j = j/len(labels)
		percent.append(j)
	# colors = np.argsort(percent)

	# plt.pie(percent, colors=np.array(centroid/255), labels=np.arange(len(centroid)))
	# plt.show()
	return (centroid, percent)


def extract_ropes(image, num, length):
	original = image.copy()
	centroids, percent = sort_colors(image)
	percent = np.array(percent)
	print(percent)
	colors = np.argsort(percent)[:2]
	ind_color1 = colors[0]
	ind_color2 = colors[1]
	print(ind_color1)
	print(ind_color2)
	color1_cent = centroids[ind_color1]
	color2_cent = centroids[ind_color2]
	color1_lower, color1_upper = color1_cent - 90, color1_cent + 90
	color2_lower, color2_upper = color2_cent - 90, color2_cent + 90

	mask1 = cv2.inRange(image, color1_lower, color1_upper)
	cnts1 = cv2.findContours(mask1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	cnts1 = cnts1[0] if len(cnts1) == 2 else cnts1[1]
	cv2.fillPoly(mask1, cnts1, (255,255,255))
	result1 = cv2.bitwise_and(original, original, mask=mask1)
	filename = '%05d.png'%(num)
	cv2.imwrite('./extracted_images/{}'.format(filename), result1)

	mask2 = cv2.inRange(image, color2_lower, color2_upper)
	cnts2 = cv2.findContours(mask2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	cnts2 = cnts2[0] if len(cnts2) == 2 else cnts2[1]
	cv2.fillPoly(mask2, cnts2, (255,255,255))
	result2 = cv2.bitwise_and(original, original, mask=mask2)
	filename = '%05d.png'%(num + length)
	cv2.imwrite('./extracted_images/{}'.format(filename), result2)

def segment_cables(img, num, length):
    Z = img.copy()
    Z = Z.reshape((-1,3))
    Z = np.float32(Z)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    K = 3
    ret,label,center=cv2.kmeans(Z,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
    #ret,label,center=cv2.kmeans(Z,K,None,criteria,10,cv2.KMEANS_PP_CENTERS)
    background_color = center[np.argmin(np.mean(center, axis=1))]
    color1, color2 = [c for c in center if list(c) != list(background_color)]
    center = np.uint8(center)
    res = center[label.flatten()]
    res2 = res.reshape((img.shape))
    kernel = (3,3)
    res2 = cv2.GaussianBlur(res2, kernel, 0)
    y1,x1,_ = np.where(res2 != np.uint8(color1))
    y2,x2,_ = np.where(res2 != np.uint8(color2))
    cable1 = img.copy()
    cable2 = img.copy()
    cable1[y1,x1] = (0,0,0)
    cable2[y2,x2] = (0,0,0)
    # cable1 = cv2.morphologyEx(cable1, cv2.MORPH_OPEN, kernel)
    # cable2 = cv2.morphologyEx(cable2, cv2.MORPH_OPEN, kernel)
    # cv2.imshow("res", np.hstack((cable1,cable2)))
    # cv2.waitKey(0)

    gray_cable1 = cv2.cvtColor(cable1, cv2.COLOR_BGR2GRAY)
    gray_cable2 = cv2.cvtColor(cable2, cv2.COLOR_BGR2GRAY)

    if cv2.countNonZero(gray_cable2) > cv2.countNonZero(gray_cable1):
    	pass
    else:
    	pass

    filename = '%05d.png'%(num)
    filename2 = '%05d.png'%(num + length)
    cv2.imwrite('./extracted_images/{}'.format(filename), cable1)
    cv2.imwrite('./extracted_images/{}'.format(filename2), cable2)

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('-d', '--dir', type=str, default='./images')
	# parser.add_argument('-m', '--masks', type=str, default="./render_kpts_output/image_masks")
	args = parser.parse_args()
	if os.path.exists('./extracted_images'):
		os.system('rm -r ./extracted_images')
	os.mkdir('./extracted_images')
	img_dir = args.dir
	# mask_dir = args.masks
	for i in range(len(os.listdir(img_dir))):
			print(os.path.join(img_dir, '%05d.png'%(i)))
			# mask = cv2.imread(os.path.join(mask_dir, '%05d.jpg'%i))
			img = cv2.imread(os.path.join(img_dir, '%05d.png'%(i)))
			segment_cables(img, i, len(os.listdir(img_dir)))
