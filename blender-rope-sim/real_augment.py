import os
import xml.etree.cElementTree as ET
from xml.dom import minidom
import random
import colorsys
import numpy as np
import cv2
import imgaug as ia
from imgaug import augmenters as iaa
from imgaug.augmentables import Keypoint, KeypointsOnImage

sometimes = lambda aug: iaa.Sometimes(0.5, aug)

KPT_AUGS = [ 
    iaa.LinearContrast((0.95, 1.05), per_channel=0.25), 
    iaa.Add((-10, 10), per_channel=False),
    iaa.GammaContrast((0.95, 1.05)),
    iaa.GaussianBlur(sigma=(0.0, 0.6)),
    iaa.MultiplySaturation((0.95, 1.05)),
    iaa.AdditiveGaussianNoise(scale=(0, 0.0125*255)),
    iaa.flip.Flipud(0.5),
    sometimes(iaa.Affine(
                scale={"x": (0.8, 1.2), "y": (0.8, 1.2)}, # scale images to 80-120% of their size, individually per axis
                translate_percent={"x": (-0.25, 0.25), "y": (-0.2, 0.2)}, # translate by -20 to +20 percent (per axis)
                rotate=(-30, 30), # rotate by -45 to +45 degrees
                shear=(-10, 10), # shear by -16 to +16 degrees
                order=[0, 1], # use nearest neighbour or bilinear interpolation (fast)
                cval=(100, 100), # if mode is constant, use a cval between 0 and 255
                mode=ia.ALL # use any of scikit-image's warping modes (see 2nd image from the top for examples)
            ))
    ]

# Uncomment for RGB
#KPT_AUGS = [ 
#    iaa.AddToHueAndSaturation((-20, 20)),
#    iaa.LinearContrast((0.95, 1.05), per_channel=0.25), 
#    iaa.Add((-10, 10), per_channel=True),
#    iaa.GammaContrast((0.95, 1.05)),
#    iaa.GaussianBlur(sigma=(0.0, 0.6)),
#    iaa.ChangeColorTemperature((3000,35000)),
#    iaa.MultiplySaturation((0.95, 1.05)),
#    iaa.AdditiveGaussianNoise(scale=(0, 0.0125*255)),
#    iaa.flip.Flipud(0.5),
#    sometimes(iaa.Affine(
#                scale={"x": (0.8, 1.2), "y": (0.8, 1.2)}, # scale images to 80-120% of their size, individually per axis
#                translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)}, # translate by -20 to +20 percent (per axis)
#                rotate=(-20, 20), # rotate by -45 to +45 degrees
#                shear=(-10, 10), # shear by -16 to +16 degrees
#                order=[0, 1], # use nearest neighbour or bilinear interpolation (fast)
#                cval=(0, 100), # if mode is constant, use a cval between 0 and 255
#                mode=ia.ALL # use any of scikit-image's warping modes (see 2nd image from the top for examples)
#            ))
#    ]

BBOX_AUGS = KPT_AUGS[:-1] + [
        iaa.flip.Fliplr(0.5),
        sometimes(iaa.Affine(
                scale={"x": (0.8, 1.2), "y": (0.8, 1.2)}, # scale images to 80-120% of their size, individually per axis
                translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)}, # translate by -20 to +20 percent (per axis)
                order=[0, 1], # use nearest neighbour or bilinear interpolation (fast)
                cval=(30, 60), # if mode is constant, use a cval between 0 and 255
                mode=ia.ALL # use any of scikit-image's warping modes (see 2nd image from the top for examples)
            ))
        ]
seq_kpts = iaa.Sequential(KPT_AUGS, random_order=True) 
seq_bbox = iaa.Sequential(BBOX_AUGS, random_order=True) 

def create_labimg_xml(xml_filename, annotation_list):
    annotation = ET.Element('annotation')
    size = ET.SubElement(annotation, 'size')
    ET.SubElement(size, 'width').text = str(640)
    ET.SubElement(size, 'height').text = str(480)
    ET.SubElement(size, 'depth').text = str(3)
    for annot in annotation_list:
        xmin, ymin, xmax, ymax = annot
        object = ET.SubElement(annotation, 'object')
        bndbox = ET.SubElement(object, 'bndbox')
        ET.SubElement(bndbox, 'xmin').text = str(xmin)
        ET.SubElement(bndbox, 'ymin').text = str(ymin)
        ET.SubElement(bndbox, 'xmax').text = str(xmax)
        ET.SubElement(bndbox, 'ymax').text = str(ymax)
    tree = ET.ElementTree(annotation)
    xmlstr = minidom.parseString(ET.tostring(annotation)).toprettyxml(indent="   ")
    with open(xml_filename, "w") as f:
        f.write(xmlstr)

def process_bbox_annots(annot_filepath, kps_aug):
    annot_list = []
    #paired_kps = kps_aug.reshape(-1,2)
    for i in range(0, len(kps_aug), 2):
        corner1 = kps_aug[i]
        corner2 = kps_aug[i+1]
        corners = np.array([corner1, corner2])
        xs, ys = corners[:,0], corners[:,1]
        min_x = min(xs[0], xs[1])
        min_y = min(ys[0], ys[1])
        max_x = max(xs[0], xs[1])
        max_y = max(ys[0], ys[1])
        annot_list.append([min_x, min_y, max_x, max_y])
    create_labimg_xml(annot_filepath, annot_list)

def augment(img, keypoints, output_dir_img, output_dir_kpt, new_idx, show=False, mode='kpt', depth_img=None, depth_output_dir_img=None):
    seq = seq_kpts if mode=='kpt' else seq_bbox
    kps = [Keypoint(x, y) for x, y in keypoints]
    kps = KeypointsOnImage(kps, shape=img.shape)
    img_aug, kps_aug = seq(image=img, keypoints=kps)
    #seq = seq.to_deterministic()
    #depth_img_aug = seq(image=depth_img)
    vis_img_aug = img_aug.copy()
    kps_aug = kps_aug.to_xy_array().astype(int)
    
    for i, (u,v) in enumerate(kps_aug):
        (r, g, b) = colorsys.hsv_to_rgb(float(i)/keypoints.shape[0], 1.0, 1.0)
        R, G, B = int(255 * r), int(255 * g), int(255 * b)
        cv2.circle(vis_img_aug,(u,v),4,(R,G,B), -1)
    if show:
        cv2.imshow("img", img_aug)
        cv2.waitKey(0)

    cv2.imwrite(os.path.join(output_dir_img, "%05d.png"%new_idx), img_aug)
    if depth_img is not None:
        cv2.imwrite(os.path.join(depth_output_dir_img, "%05d.png"%new_idx), depth_img_aug)
    if mode == 'kpt':
        np.save(os.path.join(output_dir_kpt, "%05d.npy"%new_idx), kps_aug)
    else: # BBOX
        process_bbox_annots(os.path.join(output_dir_kpt, "%05d.xml"%new_idx), kps_aug)
    

if __name__ == '__main__':
    if not os.path.exists("./annots"):
        os.mkdir('./annots')
    else:
        os.system('rm -r ./annots')
        os.mkdir('./annots')
    keypoints_dir = 'keypoints'
    bbox_dir = 'annots'

    img_dir = 'images'
    output_dir_img = img_dir

    #depth_img_dir = 'images_depth'
    #depth_output_dir_img = depth_img_dir

    idx = len(os.listdir(img_dir))
    #idx = 0
    orig_len = len(os.listdir(img_dir))
    num_augs_per = 20 #10
    mode = 'kpt'
    output_dir_annots = keypoints_dir if mode =='kpt' else bbox_dir
    if mode == 'bbox':
        for i in range(orig_len):
            img = cv2.imread(os.path.join(img_dir, '%05d.png'%i))
            kpts = np.load(os.path.join(keypoints_dir, '%05d.npy'%i))
            xml_filename = os.path.join(output_dir_annots, '%05d.xml'%i)
            process_bbox_annots(xml_filename, kpts)
    for i in range(orig_len):
        print(i, orig_len)
        img = cv2.imread(os.path.join(img_dir, '%05d.png'%i))
        #depth_img = cv2.imread(os.path.join(depth_img_dir, '%05d.jpg'%i))
        kpts = np.load(os.path.join(keypoints_dir, '%05d.npy'%i))
        for _ in range(num_augs_per):
            #augment(img, kpts, output_dir_img, output_dir_annots, idx+i, show=False, 
            #mode=mode, depth_img=depth_img, depth_output_dir_img=depth_output_dir_img)
            augment(img, kpts, output_dir_img, output_dir_annots, idx+i, show=False, mode=mode)
            idx += 1
        idx -= 1