import cv2
import numpy as np

def show_images(images):
    for i, image in enumerate(images):
        cv2.imshow(f"image{i}", image)
    while True:
        keycode = cv2.waitKey(0)
        if keycode == 27:
            break
    cv2.destroyAllWindows()

def show_image(image):
    cv2.imshow(f"image", image)
    while True:
        keycode = cv2.waitKey(0)
        if keycode == 27:
            break
    cv2.destroyAllWindows()

def check_decomposition(P, k, r, t):
    print("--------")
    print("initial P: ", P, "\n")
    print("found k: ", k, "\n")
    print("found r: ", r, "\n")
    print("found t: ", t, "\n")
    ext = np.hstack((r,t))
    print("ext mat: ", ext, "\n")
    res = k @ ext
    print("reconstructed P: ", res, "\n")

def check_image(image):
    print("--------")
    min = image.min()
    max = image.max()
    print("min: ", min, "\n")
    print("next min: ", image[image > min].min(), "\n")
    print("min > 0: ", image[image > 0.0].min(), "\n")
    print("max: ", max, "\n")
    print("next max: ", image[image < max].max(), "\n")
    
