import os
import time
import threading
import multiprocessing as mp

import keyboard
import numpy as np
from ultralytics import YOLO

import dataset
import helper
import viewer as v
import slam

start_frame = 150

######## PARAMETERS ########
viewer_enabled = True

#  stereo matcher
sad_window = 16
num_disparities = sad_window * 16
stereo_matcher_method = 'sgbm'   #'sgbm' or 'bm'

#   extractor (detector)
detector_method = 'sift' # 'sift' or 'orb'
nfeatures = 5000

#   matching
matcher_method = 'BF'  # 'FLANN' or 'BF'
lowe_ratio = 0.4

#  filtering
max_depth = 500

#  mapping
distance_threshold = 1.0  #meters




class Controller:
    def __init__(self):
        self.autoplay = False
        self.exit = False
        self.autoplay_lock = threading.Lock()
        self.exit_lock = threading.Lock()

        self.map = None

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
            if event.name == "enter":
                if self.map is not None:
                    print(f"TARGETS : {self.map.target_points}")
    
    def set_map(self, map):
        self.map = map


def yolo_process(pipe):
    yolo_model = YOLO("../yolov8/weights.pt")
    while True:
        input = pipe.recv()
        if input is None:  # Termination signal
            break
        begin = time.time()

        image_left_0 = input
        result = yolo_model.predict(image_left_0, conf = 0.8, verbose=False)[0]
        boxes = result.boxes
        probs = result.probs
        pred = result.plot()
        target_points = []
        for box in boxes.data:
            box = box.cpu()
            print(box)
            target_points.append([(box[0] + box[2])/2, (box[1] + box[3])/2])
        
        end = time.time()
        print(f" *** yolo process time: {end-begin} fps: {1/(end-begin)}")

        pipe.send(target_points)

def depth_process(pipe):
    k_l, t_l, t_r = pipe.recv()
    while True:
        input = pipe.recv()
        if input is None:  # Termination signal
            break
        begin = time.time()

        image_left_0, image_right_0 = input
        disp = slam.left_disparity_map(image_left_0, image_right_0, num_disparities=num_disparities, method=stereo_matcher_method)

        depth = slam.depth_map(disp, k_l, t_l ,t_r)

        end = time.time()
        print(f" *** depth process time: {end-begin} fps: {1/(end-begin)}")

        pipe.send(depth)

def extract_process(pipe):
    mask = pipe.recv()
    while True:
        input = pipe.recv()
        if input is None:  # Termination signal
            break
        begin = time.time()

        image = input
        kps, des = slam.extract_features(image, detector_method, mask, nfeatures=nfeatures)

        end = time.time()
        print(f" *** extract process time: {end-begin} fps: {1/(end-begin)}")

        pipe.send(([kp.pt for kp in kps], des))



if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    #### MULTIPROCESSING
    yolo_pipe, yolo_pipe_slave = mp.Pipe()
    depth_pipe, depth_pipe_slave = mp.Pipe()
    extract_pipe_0, extract_pipe_slave_0 = mp.Pipe()
    extract_pipe_1, extract_pipe_slave_1 = mp.Pipe()

    yolo_p = mp.Process(target=yolo_process, args=(yolo_pipe_slave,))
    depth_p = mp.Process(target=depth_process, args=(depth_pipe_slave,))
    extract_p_0 = mp.Process(target=extract_process, args=(extract_pipe_slave_0,))
    extract_p_1 = mp.Process(target=extract_process, args=(extract_pipe_slave_1,))
    
    yolo_p.start()
    depth_p.start()
    extract_p_0.start()
    extract_p_1.start()

    ####
    
    controller = Controller()
    keyboard.hook(controller.on_key_event)

    sequence = dataset.Kitti("00_2")
    num_frames = sequence.num_frames   #1261
    map = slam.Map(num_frames)
    controller.set_map(map)

    if viewer_enabled:
        viewer = v.Viewer()
        viewer.map = map
        viewer.start()
    else:
        viewer = None
    
    P_l = sequence.P2
    P_r = sequence.P3

    k_l, r_l, t_l = slam.decompose_projection_matrix(P_l)
    k_r, r_r, t_r = slam.decompose_projection_matrix(P_r)

    mask = slam.left_rect_mask(sequence.width, sequence.height, num_disparities)

    i = start_frame      #start frame
    
    
    just_started = True

    depth_pipe.send((k_l, t_l, t_r))
    extract_pipe_0.send(mask)
    extract_pipe_1.send(mask)

    while True:
        if not just_started:
            controller.exit_lock.acquire()
            if controller.exit:
                if viewer_enabled:
                    viewer.stop()
                yolo_pipe.send(None)
                depth_pipe.send(None)
                extract_pipe_0.send(None)
                extract_pipe_1.send(None)
                yolo_p.join()
                depth_p.join()
                extract_p_0.join()
                extract_p_1.join()
                exit()
            controller.exit_lock.release()

            controller.autoplay_lock.acquire()
            if controller.autoplay:
                controller.autoplay_lock.release()
                i = helper.clamp(i + 1, 0, num_frames-1) 
            else:
                controller.autoplay_lock.release()
                event = keyboard.read_event()
                if event.name == 'right' and event.event_type == keyboard.KEY_DOWN:
                    i = helper.clamp(i + 1, 0, num_frames-1) 
                elif event.name == 'left' and event.event_type == keyboard.KEY_DOWN:
                    i = helper.clamp(i - 1, 0, num_frames-1) 
                else:
                    continue
        else:
            just_started = False
        
        image_left_0 = sequence.get_image_left(i)

        if i < num_frames - 1 and map.frames[i].processed == False:
            
            ########################## PROCESS FRAME #################################################
            timestamps = []
            timestamps.append(time.time())  ##0

            image_right_0 = sequence.get_image_right(i)
            image_left_1 = sequence.get_image_left(i+1)

            timestamps.append(time.time())  ##1

            extract_pipe_0.send(image_left_0)
            extract_pipe_1.send(image_left_1)
            depth_pipe.send((image_left_0, image_right_0))
            yolo_pipe.send(image_left_0)

            kp_0, des_0 = extract_pipe_0.recv()
            kp_1, des_1 = extract_pipe_1.recv()

            timestamps.append(time.time())  ##2
        
            matches = slam.match_features(des_0, des_1, method=matcher_method, detector = detector_method, lowe_ratio=lowe_ratio)
            
            timestamps.append(time.time())  ##3
    
            target_points = yolo_pipe.recv()
            depth = depth_pipe.recv()

            timestamps.append(time.time())  ##4

            r_mat, t_vec, good_points, target_points_3d = slam.estimate_motion(matches, kp_0, kp_1, k_l, target_points, depth, max_depth=max_depth)

            timestamps.append(time.time())  ##5
            
            # get transformation from frame 0 to frame 1
            inv_Tr = np.eye(4)
            inv_Tr[:3, :3] = r_mat
            inv_Tr[:3, 3] = t_vec.T
            Tr = np.eye(4)
            Tr = Tr.dot(np.linalg.inv(inv_Tr))


            frame_0 = map.frames[i]
            frame_1 = map.frames[i+1]

            points_3d_global = slam.to_global(good_points, frame_0.pose)
            frame_0.points = points_3d_global
            frame_0.processed = True

            frame_1.pose = frame_0.pose @ Tr
            
            target_points_global = slam.to_global(target_points_3d, frame_0.pose)
            for target_point_global in target_points_global:
                map.append_target(target_point_global, distance_threshold)
            
            if viewer_enabled:
                viewer.update(frames = [frame_0, frame_1], targets=map.target_points)
                
            timestamps.append(time.time())  ##6

            helper.print_timestamps(timestamps)

        if viewer_enabled:
            viewer.update(state = i, image=image_left_0)
 