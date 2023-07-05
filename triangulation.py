import sys
import cv2
import numpy as np
import time


def find_depth(right_point, left_point, frame_right, frame_left, baseline, f, alpha):

    # CONVERT FOCAL LENGTH f FROM [mm] TO [pixel]:
    height_right, width_right, depth_right = frame_right.shape
    height_left, width_left, depth_left = frame_left.shape

    if width_right == width_left:
        # convert from degrees into radian, because
        f_pixel = width_right / (2 * np.tan(alpha * 0.5 * np.pi/180))

    else:
        print('Left and right camera frames do not have the same pixel width')

    x_right = right_point[0]
    x_left = left_point[0]

    # CALCULATE THE DISPARITY:
    # Displacement between left and right frames [pixels]
    disparity = x_left-x_right

    # CALCULATE DEPTH z:
    zDepth = ((baseline*f_pixel)/disparity)  # Depth in [cm]

    zDepth = "{:.2f}".format(zDepth)

    return zDepth
