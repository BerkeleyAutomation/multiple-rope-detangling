import cv2
import os
import imgaug as ia
from imgaug import augmenters as iaa
import numpy as np
aug = iaa.CoarseDropout((0.0, 0.3), size_percent=(0.03, 0.04))
def segment_rope(img, plot=False):
    # Dropout random pixels
    # img = aug.augment_image(img)
    # if plot:
    #     cv2.imshow("img", img)
    #     cv2.waitKey(0)

    # Convert to grayscale and segment rope
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray,95,255,cv2.THRESH_BINARY)
    # Get RGB segmask
    img = cv2.bitwise_and(img, img, mask=mask)
    vis = img.copy()

    img_2d = img.reshape(-1, img.shape[-1])
    non_zeros = np.where(np.any(img_2d > [0, 0, 0], axis=-1))[0]

    random_pixel = non_zeros[np.random.randint(len(non_zeros))]
    pixel = img_2d[random_pixel]
    print(pixel)

    u = random_pixel % 640
    v = random_pixel // 640
    no_drop = np.random.randint(3)
    if no_drop != 0:
        (W,H) = (13,18)
        ang = np.random.randint(-60, 60)
        P0 = (u,v-5)
        rr = RRect(P0,(W,H),ang)
        rr.draw(vis)

    if plot:
        cv2.imshow("fg", img)
        cv2.waitKey(0)
        cv2.imshow("fg", vis)
        cv2.waitKey(0)
    return vis

class RRect:
    def __init__(self, p0, s, ang):
        self.p0 = [int(p0[0]),int(p0[1])]
        (self.W, self.H) = s
        self.ang = ang
        self.p1,self.p2,self.p3 = self.get_verts(p0,s[0],s[1],ang)
        self.verts = [self.p0,self.p1,self.p2,self.p3]

    def get_verts(self, p0, W, H, ang):
        sin = np.sin(ang/180*3.14159)
        cos = np.cos(ang/180*3.14159)
        P1 = [int(self.H*sin)+p0[0],int(self.H*cos)+p0[1]]
        P2 = [int(self.W*cos)+P1[0],int(-self.W*sin)+P1[1]]
        P3 = [int(self.W*cos)+p0[0],int(-self.W*sin)+p0[1]]
        return [P1,P2,P3]

    def draw(self, image):
        for i in range(len(self.verts)-1):
            cv2.line(image, (self.verts[i][0], self.verts[i][1]), (self.verts[i+1][0],self.verts[i+1][1]), (0,0,0), 12)
        cv2.line(image, (self.verts[3][0], self.verts[3][1]), (self.verts[0][0], self.verts[0][1]), (0,0,0), 12)


if __name__ == '__main__':
    folder = 'real_images/hairtie_resized'
    output_folder = 'hairtie_resized_masks'
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)
    for f in os.listdir(folder):
        if f != '.DS_Store':
            img = cv2.imread(os.path.join(folder, f))
            m = segment_rope(img, plot=False)
            cv2.imwrite(os.path.join(output_folder, f), m)




