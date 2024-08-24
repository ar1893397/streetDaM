from multiprocessing import Process, Queue

import numpy as np
import cv2
import pangolin
import OpenGL.GL as gl



class Viewer():
    def __init__(self, w, h):
        self.w = w
        self.h = h
        self.q_image = Queue()
    
    def start(self):
        p = Process(target=self.viewer_thread) 
        p.daemon = True
        p.start()

    def viewer_thread(self):
        pangolin.CreateWindowAndBind('Main', self.w, self.h)
        gl.glPixelStorei(gl.GL_UNPACK_ALIGNMENT, 1)
        #gl.glEnable(gl.GL_DEPTH_TEST)
        self.d_img = pangolin.Display('image')
        self.d_img.SetBounds(1-(376.0/self.h), 1.0, 0.0, 1.0)  #SetBounds(bottom, top, left, right, aspect)
        #self.d_img.SetLock(pangolin.Lock.LockLeft, pangolin.Lock.LockTop)
        self.texture = pangolin.GlTexture(1241, 376, gl.GL_RGB, False, 0, gl.GL_RGB, gl.GL_UNSIGNED_BYTE)
        self.image = np.ones((376,1241, 3), 'uint8') * 255

        while not pangolin.ShouldQuit():
            self.refresh()

    def refresh(self):
        if not self.q_image.empty():
            self.image = self.q_image.get()
            if self.image.ndim == 3:
                self.image = self.image[::-1, :, ::-1]
            else:
                self.image = np.repeat(self.image[::-1, :, np.newaxis], 3, axis=2)
            self.image = cv2.resize(self.image, (1241, 376))

        self.texture.Upload(self.image, gl.GL_RGB, gl.GL_UNSIGNED_BYTE)
        self.d_img.Activate()
        gl.glColor3f(1.0, 1.0, 1.0)
        self.texture.RenderToViewport()

        pangolin.FinishFrame()

    def update_image(self, image):
        self.q_image.put(image)