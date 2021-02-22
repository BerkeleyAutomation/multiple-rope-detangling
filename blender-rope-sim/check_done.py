import cv2
import numpy as np
import argparse
from xml.etree import ElementTree
import os
import math
import json
import colorsys

#have a model that returns the 4 keypoints for an image
#make 4 copies of that image
#run analysis for each endpoint and get the pin and pull predictions
#check all 4 predictions in this script: if 1 red and 1 white return a no-op, return True for done, else False

