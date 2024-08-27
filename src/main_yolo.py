import os

from ultralytics import YOLO

import dataset
import helper


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    yolo_model = YOLO("../yolov8/weights.pt")

    sequence = dataset.Kitti("00")

    left_image = sequence.get_image_left(0)
    result = yolo_model.predict(left_image, conf = 0.8)[0]
    boxes = result.boxes
    probs = result.probs
    pred = result.plot()

    helper.show_image(pred)