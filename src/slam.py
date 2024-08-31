import math

import cv2
import numpy as np


def left_disparity_map(img_left, img_right, method = "bm", num_disparities = 16*16):
    img_left = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY)
    img_right = cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY)
    #num_disparities = sad_window*16
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

def left_rect_mask(width, height, x):
    mask = np.zeros((height, width), dtype=np.uint8)
    mask[:, x:] = 255
    return mask

def extract_features(image, method = 'sift', mask = None, nfeatures=3000):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if method == 'orb':
        extractor = cv2.ORB_create(nfeatures=nfeatures)
    if method == 'sift':
        extractor = cv2.SIFT_create(nfeatures=nfeatures)
    
    keypoints, descriptors = extractor.detectAndCompute(image, mask)

    return keypoints, descriptors


def match_features(des_0, des_1, method = 'BF', detector = 'sift', lowe_ratio = 0.5):
    if method == 'BF':
        if detector == 'sift':
            matcher = cv2.BFMatcher_create(cv2.NORM_L2, crossCheck=False)
        elif detector == 'orb':
            matcher = cv2.BFMatcher_create(cv2.NORM_HAMMING2, crossCheck=False)
        matches = matcher.knnMatch(des_0, des_1, k=2)
    elif method == 'FLANN':
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        matcher = cv2.FlannBasedMatcher(index_params, search_params)
        matches = matcher.knnMatch(des_0, des_1, k=2)
    #ratio test
    good = []
    for m,n in matches:
        if m.distance < lowe_ratio*n.distance:
            good.append(m)
    return good


def estimate_motion(match, kp1, kp2, k, target_points, depth1, max_depth=3000):
    rmat = np.eye(3)
    tvec = np.zeros((3, 1))

    image1_points = np.float32([kp1[m.queryIdx] for m in match])
    image2_points = np.float32([kp2[m.trainIdx] for m in match])

    cx = k[0, 2]
    cy = k[1, 2]
    fx = k[0, 0]
    fy = k[1, 1]
    object_points = np.zeros((0, 3))
    target_points_3d = np.zeros((0,3))
    delete = []

    for i, (u, v) in enumerate(image1_points):
        z = depth1[int(v), int(u)]

        if z > max_depth:
            delete.append(i)
            continue

        x = z*(u-cx)/fx
        y = z*(v-cy)/fy
        object_points = np.vstack([object_points, np.array([x, y, z])])

    for (u, v) in target_points:
        z = depth1[int(v), int(u)]
        if z > max_depth:
            continue
        x = z*(u-cx)/fx
        y = z*(v-cy)/fy
        target_points_3d = np.vstack([target_points_3d, np.array([x, y, z])])


    image1_points = np.delete(image1_points, delete, 0)
    image2_points = np.delete(image2_points, delete, 0)
    
    _, rvec, tvec, inliers = cv2.solvePnPRansac(object_points, image2_points, k, None)

    rmat = cv2.Rodrigues(rvec)[0]

    return rmat, tvec, object_points, target_points_3d


def to_global(points, pose):
    ones_column = np.ones((points.shape[0], 1))
    points_4d = np.hstack((points, ones_column))
    points_3d_global = points_4d @ pose.T
    return points_3d_global    

def distance(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2 + (p1[2] - p2[2]) ** 2)

class Map():
    def __init__(self, num_frames):
        self.frames = []
        self.target_points = []
        for i in range(0, num_frames):
            self.frames.append(Frame(i))

    def append_target(self, point, threshold):
        is_equivalent = any(distance(point, existing_point) < threshold for existing_point in self.target_points)
        
        if not is_equivalent:
            self.target_points.append(point)


class Frame():
    def __init__(self, i):
        self.i = i
        self.processed = False
        self.points = []
        self.pose = np.eye(4)