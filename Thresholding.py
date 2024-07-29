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

    def forward(self, img):
        """ Take an image and extract all relevant pixels.

        Parameters:
            img (np.array) : Input image

        Returns :
            binary (np.array) : A binary image represent all positions of relevant pixels.
        """
        img = cv2.GaussianBlur(img, (5, 5), 1)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        lower = np.array([0, 0, 110])
        upper = np.array([255, 255, 250])
        # mask = binary image => true(white 255), false(black 0)
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