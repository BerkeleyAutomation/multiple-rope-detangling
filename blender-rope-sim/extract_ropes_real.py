import cv2
import os
import imgaug as ia
from imgaug import augmenters as iaa
import numpy as np

def extract_rope(img, filename):
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	_, mask = cv2.threshold(gray,95,255,cv2.THRESH_BINARY)
	# Get RGB segmask
	img = cv2.bitwise_and(img, img, mask=mask)
	cv2.imwrite('./two_hairties_extracted/{}'.format(filename), img) 

def red_rope(image, filename):
	#blurring and smoothin
	hsv = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)

	#lower red
	lower_red = np.array([0,50,50])
	upper_red = np.array([10,255,255])

	#upper red
	lower_red2 = np.array([170,50,50])
	upper_red2 = np.array([180,255,255])

	mask = cv2.inRange(hsv, lower_red, upper_red)
	res = cv2.bitwise_and(image, image, mask= mask)


	mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
	res2 = cv2.bitwise_and(image, image, mask= mask2)

	img3 = res+res2
	img4 = cv2.add(res,res2)
	img5 = cv2.addWeighted(res,0.5,res2,0.5,0)


	kernel = np.ones((15,15),np.float32)/225
	smoothed = cv2.filter2D(res,-1,kernel)
	smoothed2 = cv2.filter2D(img3,-1,kernel)

	# splits = filename.split('.')
	# file = splits[0]
	# num = int(file[5:])
	# print(num)
	# filename = 'color%d.png'%(num+19)
	cv2.imwrite('{}'.format(filename), img4)


def white_rope(img, filename):
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	_, mask = cv2.threshold(gray,95,255,cv2.THRESH_BINARY)
	# Get RGB segmask
	img = cv2.bitwise_and(img, img, mask=mask)
	gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	if cv2.countNonZero(gray_img) != 0:
		cv2.imwrite('{}'.format(filename), img)


if __name__ == '__main__':
	folder = './real_images/hairtie_resized'
	output_folder = folder + '_extract'
	if not os.path.exists(output_folder):
		os.mkdir(output_folder)
	for f in os.listdir(folder):
		if f != '.DS_Store':
			img = cv2.imread(os.path.join(folder, f))
			splits = f.split('.')
			file = splits[0]
			num = int(file[5:])
			print(num)
			filename = 'color%d.png'%(num+19)
			red_rope(img, os.path.join(output_folder, filename))
			white_rope(img, os.path.join(output_folder, f))



