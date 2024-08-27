import os

import pandas as pd
import cv2

class Kitti():
    def __init__(self, sequence, is_rgb = True, poses = False):
        self.sequence = sequence
        self.is_rgb = is_rgb
        if is_rgb:
            prefix = "rgb"
        else:
            prefix = "gray"
        
        sequence_dir = f"../dataset/sequences/{prefix}/{sequence}/"        
        if is_rgb:
            self.images_left_dir = sequence_dir + 'image_2/'
            self.images_right_dir = sequence_dir + 'image_3/'
        else:
            self.images_left_dir = sequence_dir + 'image_0/'
            self.images_right_dir = sequence_dir + 'image_1/'
        self.images_left_files = sorted(os.listdir(self.images_left_dir))
        self.images_right_files = sorted(os.listdir(self.images_right_dir))

        self.num_frames = len(self.images_left_files)
        first_image = cv2.imread(self.images_left_dir + self.images_left_files[0])
        self.height = first_image.shape[0]
        self.width = first_image.shape[1]
        self.channels = first_image.shape[2]

        calib_path = sequence_dir + "calib.txt"
        calib = pd.read_csv(calib_path, delimiter=' ', header=None, index_col = 0).to_numpy().reshape(4,3,4)
        self.P0 = calib[0]
        self.P1 = calib[1]
        self.P2 = calib[2]
        self.P3 = calib[3]
        
        if poses:
            poses_path = f"../dataset/poses/{sequence}.txt"
            self.poses = pd.read_csv(poses_path, delimiter=' ', header=None).to_numpy().reshape(self.num_frames, 3, 4)

    def get_image_left_generator(self):
        return (cv2.imread(self.images_left_dir + name_left) for name_left in self.images_left_files)
        
    def get_image_right_generator(self):
        return (cv2.imread(self.images_right_dir + name_right) for name_right in self.images_right_files)

    def get_image_left(self, idx):
        return cv2.imread(self.images_left_dir + self.images_left_files[idx])
    
    def get_image_right(self, idx):
        return cv2.imread(self.images_right_dir + self.images_right_files[idx])
