from multiprocessing import Process, Queue

import numpy as np
import cv2
import pangolin
import OpenGL.GL as gl

#viewer for slam
#KITTI sequence W and H
s_w = 1241
s_h = 376

#WINDOW W and H
w_w = 1241
w_h = 1000


class Viewer():
    def __init__(self):
        self.state = None
        self.image = None   
        self.map = None     
        self.queue = Queue()

        self.points_cache = None
        self.cur_points_cache = None
        self.poses_cache = None
        self.cur_pose_cache = None
        self.targets = None
    
    def start(self):
        self.p = Process(target=self.viewer_thread) 
        self.p.daemon = True
        self.p.start()

    def stop(self):
        self.queue.put(None)
        self.p.join()

    def update_map(self, map):
        self.q_map.put((map.i, map.frames, map.target_points))

    def update(self, state = None, image = None, frames = None, targets = None):
        self.queue.put({"state": state, "image": image, "frames":frames, "targets":targets})

    def viewer_thread(self):
        pangolin.CreateWindowAndBind('Main', w_w, w_h)
        gl.glPixelStorei(gl.GL_UNPACK_ALIGNMENT, 1)
        gl.glEnable(gl.GL_DEPTH_TEST)

        
        self.d_img = pangolin.Display('image')
        self.d_img.SetBounds(1-(s_h/w_h), 1.0, 0.0, 1.0)  #SetBounds(bottom, top, left, right, aspect)
        #self.d_img.SetLock(pangolin.Lock.LockLeft, pangolin.Lock.LockTop)
        self.texture = pangolin.GlTexture(s_w, s_h, gl.GL_RGB, False, 0, gl.GL_RGB, gl.GL_UNSIGNED_BYTE)
        self.image = np.ones((s_h,s_w, 3), 'uint8') * 0  # * 255

        
        self.cam = pangolin.OpenGlRenderState(
            pangolin.ProjectionMatrix(w_w, w_h - s_h, 420, 420, w_w//2, (w_h - s_h)//2, 0.2, 10000),
            pangolin.ModelViewLookAt(0, -10, -8, 0, 0, 0, 0, -1, 0))
        self.handler = pangolin.Handler3D(self.cam)

        self.d_cam = pangolin.CreateDisplay()
        self.d_cam.SetBounds(0, 1-(s_h/w_h), 0.0, 1.0)
        self.d_cam.SetHandler(self.handler)
    
        while True:
            self.refresh()

    def refresh(self):

        if not self.queue.empty():
            input = self.queue.get()
            if input is None:
                exit()
            state = input["state"]
            image = input["image"]
            frames = input["frames"]
            targets = input["targets"]

            if state is not None:
                self.state = state

            if image is not None:
                self.image = image
                if self.image.ndim == 3:
                    self.image = self.image[::-1, :, ::-1]
                else:
                    self.image = np.repeat(self.image[::-1, :, np.newaxis], 3, axis=2)
                self.image = cv2.resize(self.image, (s_w, s_h))

            if frames is not None:
                for frame in frames:
                    self.map.frames[frame.i] = frame
            
            if targets is not None:
                self.targets = targets

            if state is not None or frames is not None:
                poses = []
                points = []
                for idx, frame in enumerate(self.map.frames):
                    if frame.processed == False:
                        continue
                    if idx == self.state:
                        self.cur_pose_cache = frame.pose
                        self.cur_points_cache = frame.points
                        continue
                    poses.append(frame.pose)
                    pt = frame.points
                    if len(pt) == 0:
                        continue
                    points.append(pt)
                self.poses_cache = np.array(poses)
                if len(points) != 0:
                    self.points_cache = np.vstack(points)

        self.d_cam.Activate(self.cam)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        gl.glClearColor(0.05, 0.05, 0.05, 1.05)


        if self.poses_cache is not None and len(self.poses_cache) != 0:
            gl.glLineWidth(1)
            gl.glColor3f(1.0, 0.0, 1.0)
            pangolin.DrawCameras(self.poses_cache)
        if self.cur_pose_cache is not None:
            gl.glLineWidth(1)
            gl.glColor3f(0.0, 1.0, 1.0)
            pangolin.DrawCameras(np.array([self.cur_pose_cache]))
            

        if self.points_cache is not None and len(self.points_cache) != 0:
            gl.glPointSize(2)
            gl.glColor3f(0.0, 0.0, 1.0)
            pangolin.DrawPoints(self.points_cache)
        if self.cur_points_cache is not None:
            gl.glPointSize(2)
            gl.glColor3f(0.0, 1.0, 1.0)
            pangolin.DrawPoints(self.cur_points_cache)
        if self.targets is not None and len(self.targets) != 0:
            gl.glPointSize(15)
            gl.glColor3f(0.0, 1.0, 0.0)
            pangolin.DrawPoints(self.targets)

        self.texture.Upload(self.image, gl.GL_RGB, gl.GL_UNSIGNED_BYTE)

        self.d_img.Activate()
        gl.glColor3f(1.0, 1.0, 1.0)
        self.texture.RenderToViewport()

        pangolin.FinishFrame()
