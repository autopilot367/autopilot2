import cv2
import numpy as np

def hist(img):
    bottom_half = img[img.shape[0]//2:, :]
    return np.sum(bottom_half, axis=0)

class LaneLines:
    """ Class containing informatoin about detected land lines.

    Attributes:
        left_fit (np.array): Coefficients of a polynomial that fit left lane line
        right_fit (np.array): Coefficients of a polynomial that fit right lane line
        parameters (dict): Dictionary containing all parameters needed for the pipeline
        debug (boolean): Flag for debug/normal mode
    """
    def __init__(self):
        """Init LaneLines.

        Parameters:
            left_fit (np.array): Coefficients of a polynomial that fit left lane line
            right_fit (np.array): Coefficients of a polynomial that fit right lane line
            binary (np.array): binary image
        """
        self.left_fit = []
        self.right_fit = []
        self.binary = None
        self.nonzero = []
        self.nonzerox = []
        self.nonzeroy = []
        self.clear_visibility = True
        self.dir = []

        self.nwindows = 12
        self.margin = 20
        self.minpix = 50

    def extract_features(self, img):
        """ Extract features from a binary image

        Parameters:
            img (np.array): A binary image
        """
        self.img = img
        self.window_height = np.int32(img.shape[0]//self.nwindows)
        self.nonzero = img.nonzero()
        self.nonzerox = np.array(self.nonzero[1])
        self.nonzeroy = np.array(self.nonzero[0])

    def forward(self, img):
        """Take a image and detect lane lines.

        Parameters:
            img (np.array): An binary image containing relevant pixels

        Returns:
            Image (np.array): An RGB image containing lane lines pixels and other details
        """
        self.extract_features(img)
        return self.fit_poly(img)

    def pixels_in_window(self, center, margin, height, img):
        """ Return all pixel that in a specific window

        Parameters:
            center (tuple): coordinate of the center of the window
            margin (int): half width of the window
            height (int): height of the window

        Returns:
            pixelx (np.array): x coordinates of pixels that lie inside the window
            pixely (np.array): y coordinates of pixels that lie inside the window
        """
        topleft = (center[0] - margin, center[1] - height // 2)
        bottomright = (center[0] + margin, center[1] + height // 2)

        condx = (topleft[0] <= self.nonzerox) & (self.nonzerox <= bottomright[0])
        condy = (topleft[1] <= self.nonzeroy) & (self.nonzeroy <= bottomright[1])
        # print(self.nonzeroy)
        # print(condy)
        # print(condx.shape)
        # print(condy.shape)
        cv2.rectangle(img, topleft, bottomright, (255, 0, 0), 2)
        cv2.imshow("sliding windows", img)
        return self.nonzerox[condx & condy], self.nonzeroy[condx & condy]

    def find_lane_pixels(self, img):
        """Find lane pixels from a binary warped image.

        Parameters:
            img (np.array): A binary warped image

        Returns:
            leftx (np.array): x coordinates of left lane pixels
            lefty (np.array): y coordinates of left lane pixels
            rightx (np.array): x coordinates of right lane pixels
            righty (np.array): y coordinates of right lane pixels
            out_img (np.array): A BGR image that use to display result later on.
        """
        assert(len(img.shape) == 2) #debugging

        out_img = np.dstack((img, img, img))

        histogram = hist(img)
        midpoint = histogram.shape[0]//2
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint

        leftx_current = leftx_base
        rightx_current = rightx_base
        y_current = img.shape[0] + self.window_height//2

        leftx, lefty, rightx, righty = [], [], [], []

        for _ in range(self.nwindows):
            y_current -= self.window_height
            center_left = (leftx_current, y_current)
            center_right = (rightx_current, y_current)

            good_left_x, good_left_y = self.pixels_in_window(center_left, self.margin, self.window_height, out_img)
            good_right_x, good_right_y = self.pixels_in_window(center_right, self.margin, self.window_height, out_img)

            leftx.extend(good_left_x)
            lefty.extend(good_left_y)
            rightx.extend(good_right_x)
            righty.extend(good_right_y)

            if len(good_left_x) > self.minpix:
                leftx_current = np.int32(np.mean(good_left_x))
            if len(good_right_x) > self.minpix:
                rightx_current = np.int32(np.mean(good_right_x))

        return leftx, lefty, rightx, righty, out_img

    def fit_poly(self, img):
        """Find the lane line from an image and draw it.

                Parameters:
                    img (np.array): a binary warped image

                Returns:
                    out_img (np.array): a BGR image that have lane line drawn on that.
                """

        leftx, lefty, rightx, righty, out_img = self.find_lane_pixels(img)
        # print(f"leftx: {leftx}")
        # print(f"lefty: {lefty}")
        # print(f"rightx: {rightx}")
        # print(f"righty: {righty}")
        # print(f"out_img: {out_img}")

        if len(lefty) > 1500:
            self.left_fit = np.polyfit(lefty, leftx, 2)
        if len(righty) > 1500:
            self.right_fit = np.polyfit(righty, rightx, 2)

        # Generate x and y values for plotting
        maxy = img.shape[0] - 1
        miny = img.shape[0] // 3
        if len(lefty):
            maxy = max(maxy, np.max(lefty))
            miny = min(miny, np.min(lefty))

        if len(righty):
            maxy = max(maxy, np.max(righty))
            miny = min(miny, np.min(righty))

        ploty = np.linspace(miny, maxy, img.shape[0])
        # print(f"self.left_fit: {self.left_fit}")
        # print(f"ploty: {ploty}")
        left_fitx = self.left_fit[0] * ploty ** 2 + self.left_fit[1] * ploty + self.left_fit[2]
        right_fitx = self.right_fit[0] * ploty ** 2 + self.right_fit[1] * ploty + self.right_fit[2]

        # Visualization
        out_img2 = np.zeros_like(out_img)
        for i, y in enumerate(ploty):
            l = int(left_fitx[i])
            r = int(right_fitx[i])
            y = int(y)
            cv2.circle(out_img2, (l, y), 5, (255, 0, 255), -1)
            cv2.circle(out_img2, (r, y), 5, (255, 0, 255), -1)
            cv2.line(out_img2, (l, y), (r, y), (0, 255, 0))

        return out_img2

