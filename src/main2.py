import os

import dataset
import slam
import helper

import numpy as np


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    sequence = dataset.Kitti("00")
    #frame 002080
    left_image = next(sequence.images_left)
    right_image = next(sequence.images_right)

    disp = slam.left_disparity_map(left_image, right_image)
    #helper.check_image(disp)

    Pl = sequence.P2
    Pr = sequence.P3

    kl, rl, tl = slam.decompose_projection_matrix(Pl)
    kr, rr, tr = slam.decompose_projection_matrix(Pr)

    depth = slam.depth_map(disp, kl, tl ,tr)
    #helper.check_image(depth)









