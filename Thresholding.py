import cv2
import numpy as np

def threshold_rel(img, lo, hi):
    vmin = np.min(img)
    vmax = np.max(img)

    vlo = vmin + (vmax - vmin) * lo
    vhi = vmin + (vmax - vmin) * hi
    return np.uint8((img >= vlo) & (img <= vhi)) * 255

def threshold_abs(img, lo, hi):
    return np.uint8((img >= lo) & (img <= hi)) * 255

class Thresholding:
    """ This class is for extracting relevant pixels in an image """
    def __init__(self):
        """ Init Thresholding """
        pass

    # def forward(self, img):
    #     hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    #     lower_white = np.array([0, 200, 100])
    #     upper_white = np.array([255, 255, 255])
    #     mask_white = cv2.inRange(hls, lower_white, upper_white)
    #
    #     lower_blue = np.array([90, 120, 70])
    #     upper_blue = np.array([130, 255, 255])
    #     mask_yellow = cv2.inRange(hls, lower_blue, upper_blue)
    #
    #     mask = cv2.bitwise_or(mask_white, mask_yellow)
    #     return mask

    def forward(self, img):
        """ Take an image and extract all relevant pixels.

        Parameters:
            img (np.array) : Input image

        Returns :
            binary (np.array) : A binary image represent all positions of relevant pixels.
        """
        img = cv2.GaussianBlur(img, (5, 5), 1)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        # h, s, v = cv2.split(hsv)
        # v_gaussian = cv2.adaptiveThreshold(v, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 1271, 2)
        lower = np.array([0, 0, 110])
        upper = np.array([255, 255, 250])
        # mask = v_gaussian
        mask = cv2.inRange(hsv, lower, upper)

        # right_lane = threshold_rel(v, 0.5, 1.0)
        # right_lane[:, :400] = 0
        # cv2.imshow("right_lane", right_lane)
        # left_lane = threshold_rel(v, 0.5, 1.0)
        # left_lane[:, 200:] = 0
        # print(f"left_lane:{left_lane}")
        # img2 = left_lane | right_lane
        # print(f"img2:{img2}")
        return mask