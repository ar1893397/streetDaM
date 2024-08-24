import os

import pandas as pd
import cv2

class Kitti():
    def __init__(self, sequence, rgb = True):
        self.sequence = sequence
        if rgb:
            prefix = "rgb"
        else:
            prefix = "gray"
        
        self.seq_dir = f"../dataset/sequences/{prefix}/{sequence}/"
        #self.poses_path = f"../dataset/poses/{sequence}.txt"
        self.calib_path = self.seq_dir + "calib.txt"
        
        if rgb:
            self.left_images_dir = self.seq_dir + 'image_2/'
            self.right_images_dir = self.seq_dir + 'image_3/'
        else:
            self.left_images_dir = self.seq_dir + 'image_0/'
            self.right_images_dir = self.seq_dir + 'image_1/'

        self.left_image_files = sorted(os.listdir(self.left_images_dir))
        self.right_image_files = sorted(os.listdir(self.right_images_dir))
        self.num_frames = len(self.left_image_files)

        #self.gt = pd.read_csv(self.poses_path, delimiter=' ', header=None).to_numpy().reshape(self.num_frames, 3, 4)

        calib = pd.read_csv(self.calib_path, delimiter=' ', header=None, index_col = 0).to_numpy().reshape(4,3,4)
        self.P0 = calib[0]
        self.P1 = calib[1]
        self.P2 = calib[2]
        self.P3 = calib[3]

        self.reset()

    def reset(self):
        self.images_left = (cv2.imread(self.left_images_dir + name_left) for name_left in self.left_image_files)
        self.images_right = (cv2.imread(self.right_images_dir + name_right) for name_right in self.right_image_files)