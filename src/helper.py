import cv2
import numpy as np

def clamp(value, min_value, max_value):
    return max(min_value, min(value, max_value))

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
    
def check_dataset(dataset, round=True):
    if round == True:
        np.set_printoptions(suppress=True, precision=4)
    print("--------")
    print(f"sequence: {dataset.sequence}\n")
    print(f"is rgb: {dataset.is_rgb}\n")
    print(f"num_frames: {dataset.num_frames}\n")
    print(f"height: {dataset.height}\n")
    print(f"width: {dataset.width}\n")
    print(f"channels: {dataset.channels}\n")
    print(f"P0: {dataset.P0}\n shape: {dataset.P0.shape} \n type: {type(dataset.P0)}\n value type: {type(dataset.P0[0][0])} \n")
    print(f"P1: {dataset.P1}\n")
    print(f"P2: {dataset.P2}\n")
    print(f"P3: {dataset.P3}\n")
    print(f"poses: {dataset.poses}\n shape: {dataset.poses.shape} \n type: {type(dataset.poses)}\n value type: {type(dataset.poses[0][0][0])} \n")
    np.set_printoptions()
    

def print_timestamps(ts):
    print(f" --- image preparation: {ts[1]-ts[0]}  --- fps: {1/(ts[1]-ts[0])}")
    print(f" --- matching: {ts[3]-ts[2]}  --- fps: {1/(ts[3]-ts[2])}")
    print(f" --- estimate motion: {ts[5]-ts[4]}  --- fps: {1/(ts[5]-ts[4])}")
    print(f" --- TOTAL TIME: {ts[6]-ts[0]}  --- fps: {1/(ts[6]-ts[0])}")
    print()
