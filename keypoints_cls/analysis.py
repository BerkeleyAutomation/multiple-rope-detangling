import pickle
import cv2
import os
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from config import *
from src.model import KeypointsGauss
#from src.model_multi_headed import KeypointsGauss
from src.dataset import KeypointsDataset, transform
from src.prediction import Prediction
from datetime import datetime
from PIL import Image
import numpy as np
import torchvision.models as models

# model
keypoints = models.resnet34(pretrained=False, num_classes=1).cuda() 
#keypoints = KeypointsGauss(NUM_KEYPOINTS, img_height=IMG_HEIGHT, img_width=IMG_WIDTH)
keypoints.load_state_dict(torch.load('checkpoints/end_condition/model_2_1_20_0.005372296337459399.pth'))

# cuda
use_cuda = torch.cuda.is_available()
#use_cuda = False
if use_cuda:
    torch.cuda.set_device(0)
    keypoints = keypoints.cuda()

prediction = Prediction(keypoints, NUM_KEYPOINTS, IMG_HEIGHT, IMG_WIDTH, use_cuda)
transform = transform = transforms.Compose([
    transforms.ToTensor()
])

image_dir = 'data/real_images/end_condition_test'

classes = {0: "Undo", 1:"Reidemeister", 2:"Terminate"}
for i, f in enumerate(sorted(os.listdir(image_dir))):
    img = cv2.imread(os.path.join(image_dir, f))
    img_t = transform(img)
    img_t = img_t.cuda()
    # GAUSS
    value = prediction.predict(img_t)
    value = value.detach().cpu().numpy()
    prediction.sort(img, value, image_id=i)
 
    #heatmap, cls = prediction.predict(img_t)
    #cls = torch.argmax(cls).item()
    #heatmap = heatmap.detach().cpu().numpy()
    #prediction.plot(img, heatmap, image_id=i, cls=cls, classes=classes)
