from multiprocessing import Process, Queue

import numpy as np
import cv2
import pangolin
import OpenGL.GL as gl

#viewer for slam

class Viewer():
    def __init__(self, w, h, sequence):
        self.sequence = sequence
        self.w = w
        self.h = h
        
        self.points = []
        self.poses = []
        self.cur_pose = None
        self.cur_points = None

        self.q_quit = Queue()
        self.q_map = Queue()
    
    def start(self):
        self.p = Process(target=self.viewer_thread) 
        self.p.daemon = True
        self.p.start()

    def stop(self):
        self.q_quit.put(1)
        self.p.join()

    def update_map(self, map):
        self.q_map.put((map.i, map.frames))

    def viewer_thread(self):
        pangolin.CreateWindowAndBind('Main', self.w, self.h)
        gl.glPixelStorei(gl.GL_UNPACK_ALIGNMENT, 1)
        gl.glEnable(gl.GL_DEPTH_TEST)

        
        self.d_img = pangolin.Display('image')
        self.d_img.SetBounds(1-(376.0/self.h), 1.0, 0.0, 1.0)  #SetBounds(bottom, top, left, right, aspect)
        #self.d_img.SetLock(pangolin.Lock.LockLeft, pangolin.Lock.LockTop)
        self.texture = pangolin.GlTexture(1241, 376, gl.GL_RGB, False, 0, gl.GL_RGB, gl.GL_UNSIGNED_BYTE)
        self.image = np.ones((376,1241, 3), 'uint8') * 255

        
        self.cam = pangolin.OpenGlRenderState(
            pangolin.ProjectionMatrix(self.w, self.h - 376, 420, 420, self.w//2, (self.h - 376)//2, 0.2, 10000),
            pangolin.ModelViewLookAt(0, -10, -8, 0, 0, 0, 0, -1, 0))
        self.handler = pangolin.Handler3D(self.cam)

        self.d_cam = pangolin.CreateDisplay()
        self.d_cam.SetBounds(0, 1-(376.0/self.h), 0.0, 1.0)
        self.d_cam.SetHandler(self.handler)
    
        #while not pangolin.ShouldQuit():
        while True:
            if not self.q_quit.empty():
                _ = self.q_quit.get()
                if not self.q_map.empty():
                    _ = self.q_map.get()
                break 
            self.refresh()

    def refresh(self):

        if not self.q_map.empty():

            i, frames = self.q_map.get()
            
            self.image = self.sequence.get_image_left(i)
            if self.image.ndim == 3:
                self.image = self.image[::-1, :, ::-1]
            else:
                self.image = np.repeat(self.image[::-1, :, np.newaxis], 3, axis=2)
            self.image = cv2.resize(self.image, (1241, 376))

            poses = []
            points = []
            for idx, frame in enumerate(frames):
                if frame.processed == False:
                    continue
                if idx == i:
                    self.cur_pose = frame.pose
                    self.cur_points = frame.points
                    continue
                poses.append(frame.pose)
                pt = frame.points
                if len(pt) == 0:
                    continue
                points.append(frame.points)
            self.poses = np.array(poses)
            if len(points) != 0:
                self.points = np.vstack(points)
            self.i = i

        self.d_cam.Activate(self.cam)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        gl.glClearColor(1.0, 1.0, 1.0, 1.0)


        if len(self.poses) != 0:
            gl.glLineWidth(1)
            gl.glColor3f(0.0, 1.0, 0.0)
            pangolin.DrawCameras(self.poses)
        if self.cur_pose is not None:
            gl.glLineWidth(1)
            gl.glColor3f(0.0, 0.0, 1.0)
            pangolin.DrawCameras(np.array([self.cur_pose]))
            

        if len(self.points) != 0:
            gl.glPointSize(2)
            gl.glColor3f(1.0, 0.0, 0.0)
            pangolin.DrawPoints(self.points)
        if self.cur_points is not None:
            gl.glPointSize(2)
            gl.glColor3f(0.0, 0.0, 1.0)
            pangolin.DrawPoints(self.cur_points)

        self.texture.Upload(self.image, gl.GL_RGB, gl.GL_UNSIGNED_BYTE)

        self.d_img.Activate()
        gl.glColor3f(1.0, 1.0, 1.0)
        self.texture.RenderToViewport()

        pangolin.FinishFrame()
