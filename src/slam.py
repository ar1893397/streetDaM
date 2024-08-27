import cv2
import numpy as np

def left_disparity_map(img_left, img_right, method = "sgbm", num_disparities = 16*16):
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

def extract_features(image, method = 'sift', mask = None):
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


def estimate_motion(match, kp1, kp2, k, depth1=None, max_depth=3000):
    rmat = np.eye(3)
    tvec = np.zeros((3, 1))
    
    image1_kps = [kp1[m.queryIdx] for m in match]
    image2_kps = [kp2[m.trainIdx] for m in match]

    image1_points = np.float32([kp.pt for kp in image1_kps])
    image2_points = np.float32([kp.pt for kp in image2_kps])

    cx = k[0, 2]
    cy = k[1, 2]
    fx = k[0, 0]
    fy = k[1, 1]
    object_points = np.zeros((0, 3))
    delete = []

    for i, (u, v) in enumerate(image1_points):
        z = depth1[int(v), int(u)]

        if z > max_depth:
            delete.append(i)
            continue

        x = z*(u-cx)/fx
        y = z*(v-cy)/fy
        object_points = np.vstack([object_points, np.array([x, y, z])])

    image1_points = np.delete(image1_points, delete, 0)
    image2_points = np.delete(image2_points, delete, 0)
    image1_kps = np.delete(image1_kps, delete, 0)
    image2_kps = np.delete(image2_kps, delete, 0)
    
    _, rvec, tvec, inliers = cv2.solvePnPRansac(object_points, image2_points, k, None)

    rmat = cv2.Rodrigues(rvec)[0]

    return rmat, tvec, image1_kps, image2_kps, object_points



class Map():
    def __init__(self, i):
        self.i = i   #state
        self.frames = []

class Frame():
    def __init__(self, i):
        self.i = i
        self.processed = False
        self.points = []
        self.pose = np.eye(4)