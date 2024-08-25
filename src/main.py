import os

import cv2
from ultralytics import YOLO

import dataset
import viewer as v


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    yolo_model = YOLO("../yolov8/weights.pt")

    viewer = v.Viewer(1241, 1000)
    viewer.start()

    sequence = dataset.Kitti("00")

    for i in range(0, sequence.num_frames):
        left_image = next(sequence.images_left)

        result = yolo_model.predict(left_image, conf = 0.8)[0]
        boxes = result.boxes
        probs = result.probs
        pred = result.plot()

        viewer.update_image(pred)