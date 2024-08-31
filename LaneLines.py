import cv2
import numpy as np
import math

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
        self.left_fit = [0, 0, 0]
        self.right_fit = [0, 0, 0]
        self.binary = None
        self.nonzero = []
        self.nonzerox = []
        self.nonzeroy = []
        self.clear_visibility = True
        self.dir = []
        
        self.radius_of_curvature = None
        self.road_inf = None
        self.curvature = None
        self.deviation = None

        self.nwindows = 12
        self.margin = 20
        self.minpix = 50

    def extract_features(self, img):
        """ Extract features from a binary image

        Parameters:
            img (np.array): A binary image
        """
        img[:180,:] = 0
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
        img, road_info, left_fitx, right_fitx, ploty = self.fit_poly(img)
        return img, road_info, left_fitx, right_fitx, ploty

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
        if np.sum(histogram) == 0:
            print("없음")
            return None, None, None, None, out_img
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
        if leftx is None:
            return out_img, None, None, None, None
        if len(lefty) > 1000:
            self.left_fit = np.polyfit(lefty, leftx, 2)
        if len(righty) > 1000:
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
            cv2.circle(out_img2, (l, y), 7, (255, 0, 255), -1)
            cv2.circle(out_img2, (r, y), 7, (255, 0, 255), -1)
            #cv2.line(out_img2, (l, y), (r, y), (0, 255, 0))
    


        """Road info"""
        #curvature
        ploty = np.linspace(0, self.img.shape[0] - 1, self.img.shape[0])
        self.ym_per_pix = 30 / self.img.shape[0]  # meters per pixel in y dimension
        self.xm_per_pix = 3.7 / self.img.shape[1] # 30 & 3.7 => arbitrary constant

        y_eval = np.max(ploty)

        if self.left_fit[0] == 0:
            left_curverad = 0
        else: left_curverad = ((1 + (2*self.left_fit[0]*y_eval*self.ym_per_pix + self.left_fit[1])**2)**1.5) / np.absolute(2*self.left_fit[0])
        if self.right_fit[0] == 0:
            right_curverad = 0
        else: right_curverad = ((1 + (2*self.right_fit[0]*y_eval*self.ym_per_pix + self.right_fit[1])**2)**1.5) / np.absolute(2*self.right_fit[0])
        
        radius_of_curvature = (left_curverad + right_curverad) / 2
        curvature = {
            'left_curvature': left_curverad,'right_curvature': right_curverad}
        # print(f"Curvature: {curvature}")
        

        left_startx, right_startx = left_fitx[len(left_fitx) - 1], right_fitx[len(right_fitx) - 1] # bottom component
        left_endx, right_endx = left_fitx[0], right_fitx[0] # top components

        direction = ((left_endx - left_startx) + (right_endx - right_startx)) / 2
        # print("direction :", direction)

        if radius_of_curvature > 2000 and abs(direction) < 100 :
            road_info = 'No Curve'
            curvature = -1
        elif radius_of_curvature <= 2000 and direction < -50 :
            road_info = 'Left Curve'
        elif radius_of_curvature <= 2000 and direction > 50 :
            road_info = 'Right Curve'
        else :
            road_info = 'Undetectable'

        #deviation
        center_lane = (right_startx + left_startx) / 2
        lane_width = right_startx - left_startx

        center_car = 720 / 2 + 12

        if lane_width == 0:
            return out_img, None, None, None, None

        pix_deviation = round((center_lane - center_car) / (lane_width/2), 6)
        deviation = round((3.5)*pix_deviation / lane_width, 2)

        if center_lane > center_car :
            deviation_state = f'Left {str(abs(deviation))}m'
        elif center_lane < center_car :
            deviation_state = f'Right {str(abs(deviation))}m'
        else :
            deviation_state = 'Center'

        #steering
        
        if radius_of_curvature > 2000:
            steering_angle = 0
        else:
            wheelbase = 2700 #mm
            steering_ratio = 20000
            # adjusted_radius = radius_of_curvature - deviation
            adjusted_radius = deviation
            if deviation == 0:
                steering_angle = 0
            else:
                # 조향각 계산
                steering_angle_rad = math.atan(wheelbase / adjusted_radius)
                steering_angle_deg = math.degrees(steering_angle_rad)

                # 조향비 적용
                steering_angle = round(steering_angle_deg * steering_ratio, 2)

        road_info = [radius_of_curvature, road_info, deviation_state, steering_angle, deviation]
        print(road_info)
        return out_img2, road_info, left_fitx, right_fitx, ploty
        