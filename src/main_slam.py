import keyboard
import os
import time
import threading

import numpy as np
import cv2

import dataset
import helper
import viewer as v
import slam

# PARAMETERS #
viewer_enabled = True

sad_window = 7
num_disparities = sad_window * 16

class Controller:
    def __init__(self):
        self.autoplay = False
        self.exit = False
        self.autoplay_lock = threading.Lock()
        self.exit_lock = threading.Lock()

    def on_key_event(self, event):
        if event.event_type == keyboard.KEY_DOWN:
            if event.name == "space":
                self.autoplay_lock.acquire()
                self.autoplay = not self.autoplay
                self.autoplay_lock.release()
            if event.name == "esc":
                self.exit_lock.acquire()
                self.exit = True 
                self.exit_lock.release()  

def main():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    controller = Controller()
    keyboard.hook(controller.on_key_event)

    sequence = dataset.Kitti("00_2")
    num_frames = sequence.num_frames

    if viewer_enabled:
        viewer = v.Viewer(1241, 1000, sequence)
        viewer.start()
    else:
        viewer = None
    
    P_l = sequence.P2
    P_r = sequence.P3

    k_l, r_l, t_l = slam.decompose_projection_matrix(P_l)
    k_r, r_r, t_r = slam.decompose_projection_matrix(P_r)

    mask = slam.left_rect_mask(sequence.width, sequence.height, num_disparities)

    i = 0          #start frame
    map = slam.Map(i)
    for idx in range(0, num_frames-1):
        map.frames.append(slam.Frame(idx))
    
    just_started = True

    while True:
        if not just_started:
            controller.exit_lock.acquire()
            if controller.exit:
                if viewer_enabled:
                    viewer.stop()
                exit()
            controller.exit_lock.release()

            controller.autoplay_lock.acquire()
            if controller.autoplay:
                controller.autoplay_lock.release()
                i = helper.clamp(i + 1, 0, num_frames) 
            else:
                controller.autoplay_lock.release()
                event = keyboard.read_event()
                if event.name == 'right' and event.event_type == keyboard.KEY_DOWN:
                    i = helper.clamp(i + 1, 0, num_frames) 
                elif event.name == 'left' and event.event_type == keyboard.KEY_DOWN:
                    i = helper.clamp(i - 1, 0, num_frames) 
                else:
                    continue
        else:
            just_started = False
        
        map.i = i

        if map.frames[i].processed == False and i != num_frames-1:
            process_frame(sequence, i, k_l, t_l, t_r, mask, map, viewer)

        if viewer_enabled:
        #updates
            #viewer.update_image(res)
            viewer.update_map(map)
        


def process_frame(sequence, i, k_l, t_l, t_r, mask, map, viewer=None):
    begin = time.time()
    image_left_0 = sequence.get_image_left(i)
    image_right_0 = sequence.get_image_right(i)

    image_left_1 = sequence.get_image_left(i+1)

        
    disp = slam.left_disparity_map(image_left_0, image_right_0, num_disparities=num_disparities)

    depth = slam.depth_map(disp, k_l, t_l ,t_r)
    
    kp_0, des_0 = slam.extract_features(image_left_0, "sift", mask)
    kp_1, des_1 = slam.extract_features(image_left_1, "sift", mask)       

    matches = slam.match_features(des_0, des_1)

    r_mat, t_vec, good_kp_0, good_kp_1, good_points = slam.estimate_motion(matches, kp_0, kp_1, k_l, depth)

    inv_Tr = np.eye(4)
    inv_Tr[:3, :3] = r_mat
    inv_Tr[:3, 3] = t_vec.T
    Tr = np.eye(4)
    Tr = Tr.dot(np.linalg.inv(inv_Tr))
    ones_column = np.ones((good_points.shape[0], 1))


    frame_0 = map.frames[i]
    frame_1 = map.frames[i+1]
    ones_column = np.ones((good_points.shape[0], 1))
    points_4d = np.hstack((good_points, ones_column))
    points_3d_global = frame_0.pose @ points_4d.T
    frame_0.points = points_3d_global.T
    frame_1.pose = frame_0.pose @ Tr
    frame_0.processed = True
    

    end = time.time()
    print(f"{i} --- time: {end-begin}  --- fps: {1/(end-begin)}")


if __name__ == "__main__":
    main()    

 