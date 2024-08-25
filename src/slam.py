import cv2
import numpy as np

def left_disparity_map(img_left, img_right, method = "sgbm"):
    img_left = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY)
    img_right = cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY)

    sad_window = 16
    num_disparities = sad_window*16
    block_size = 9

    if method == 'bm':
        matcher = cv2.StereoBM_create(numDisparities=num_disparities,
                                      blockSize=block_size
                                     )
    elif method == 'sgbm':
        matcher = cv2.StereoSGBM_create(numDisparities=num_disparities,
                                        minDisparity=0,
                                        blockSize=block_size,
                                        P1 = 8 * 1 * block_size ** 2,
                                        P2 = 32 * 1 * block_size ** 2,
                                        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
                                       )

    disp_left = matcher.compute(img_left, img_right).astype(np.float32)/16

    return disp_left


def decompose_projection_matrix(p):
    k, r, t, _, _, _, _ = cv2.decomposeProjectionMatrix(p)
    t = -(t / t[3])[:3]    #t is always flipped

    return k, r, t


def depth_map(disp_left, k_left, tl, tr):
    b = tl[0] - tr[0]
    f = k_left[0][0]
    disp_left[disp_left <= 0.0] = 0.1  #min > 0.0 = 0.5625
    depth_map = f * b /disp_left

    return depth_map




def extract_features(image, method, mask = None):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if method == 'orb':
        extractor = cv2.ORB_create() #nfeatures=5000
    if method == 'sift':
        extractor = cv2.SIFT_create()
    
    keypoints, descriptors = extractor.detectAndCompute(image, mask)

    return keypoints, descriptors


def match_features(des_0, des_1):
    #matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des_0,des_1,k=2)
    #ratio test
    good = []
    for m,n in matches:
        if m.distance < 0.5*n.distance:
            good.append(m)
    return good