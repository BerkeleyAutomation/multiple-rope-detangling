import cv2
import os
import argparse
import json
import numpy as np

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dir', type=str, default='train_sets/multiple_blackout/test')
    args = parser.parse_args()
    if os.path.exists('./reordered'):
        os.system('rm -r ./reordered')
    os.mkdir('./reordered')
    os.mkdir('./reordered/blacked_out')
    os.mkdir('./reordered/keypoints')
# 
    i = 0
    dir_len = len(os.listdir('train_sets/multiple_blackout/test/blacked_out'))
    for l in range(dir_len//3):
        for j in range(3):
            filename = "%05d_%01d_blacked_out.png"%(l,j)
            print("Relabeling " + filename)
            num = int(filename[:5])
            keypoints_filename = "%05d_%01d.npy"%(l,j)
            save_keypoints_filename = "%05d.npy"%i
            save_img_filename = '%05d.png'%i
            new_knots_info = np.load("./%s/keypoints/%s"%(args.dir, keypoints_filename))
            img = cv2.imread('./%s/blacked_out/%s'%(args.dir, filename)).copy()
            np.save("./reordered/keypoints/{}".format(save_keypoints_filename), new_knots_info)
            cv2.imwrite('./reordered/blacked_out/{}'.format(save_img_filename), img)
            i += 1
