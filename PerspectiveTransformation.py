import cv2
import numpy as np

class PerspectiveTransformation:
    """ This class is for transforming image between front view and top view

    Attributes:
        src (np.array): Coordinates of 4 source points
        dst (np.array): Coordinates of 4 destination points
        M (np.array) : Matrix to transform image from front view to top view
        M_inv (np.array) : Matrix to transform image from top view to front view
    """
    def __init__(self):
        """Init PerspectiveTransformation."""
        self.src = np.float32([(220, 230),
                               (400, 230),
                               (640, 360-5),
                               (0, 360-5)])
        self.dst = np.float32([(0, 0),
                               (640, 0),
                               (640, 360),
                               (0, 360)])
        self.M = cv2.getPerspectiveTransform(self.src, self.dst)
        self.M_inv = cv2.getPerspectiveTransform(self.dst, self.src)

    def forward(self, img, img_size=(640,360), flags=cv2.INTER_LINEAR):
        """ Take a front view image and transform to top view

        Parameters:
                   img (np.array): A front view image
                   img_size (tuple): Size of the image (width, height)
                   flags : flag to use in cv2.warpPerspective()

        Returns:
                   Image (np.array): Top view image
        """
        return cv2.warpPerspective(img, self.M, img_size, flags=flags)

    def backward(self, img, img_size=(640,360), flags=cv2.INTER_LINEAR):
        """ Take a front view image and transform to top view

        Parameters:
                   img (np.array): A front view image
                   img_size (tuple): Size of the image (width, height)
                   flags : flag to use in cv2.warpPerspective()

        Returns:
                   Image (np.array): Top view image
        """
        return cv2.warpPerspective(img, self.M_inv, img_size, flags=flags)