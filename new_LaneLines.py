import cv2
import numpy as np
import math
import matplotlib as plt

class new_LaneLines:
    def __init__(self):
        self.left_fit_hist = np.array([])
        self.right_fit_hist = np.array([])
        self.prev_left_fit = np.array([])
        self.prev_right_fit = np.array([])

    def find_lane_pixels_using_histogram(self, binary_warped):
        # Take a histogram of the bottom half of the image
        histogram = np.sum(binary_warped[binary_warped.shape[0] // 2:, :], axis=0)

        # Find the peak of the left and right halves of the histogram
        # These will be the starting point for the left and right lines
        midpoint = int(histogram.shape[0] // 2)
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint

        # Choose the number of sliding windows
        nwindows = 9
        # Set the width of the windows +/- margin
        margin = 50
        # Set minimum number of pixels found to recenter window
        minpix = 20

        # Set height of windows - based on nwindows above and image shape
        window_height = int(binary_warped.shape[0] // nwindows)
        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Current positions to be updated later for each window in nwindows
        leftx_current = leftx_base
        rightx_current = rightx_base

        # Create empty lists to receive left and right lane pixel indices
        left_lane_inds = []
        right_lane_inds = []

        # Step through the windows one by one
        for window in range(nwindows):
            # Identify window boundaries in x and y (and right and left)
            win_y_low = binary_warped.shape[0] - (window + 1) * window_height
            win_y_high = binary_warped.shape[0] - window * window_height
            win_xleft_low = leftx_current - margin
            win_xleft_high = leftx_current + margin
            win_xright_low = rightx_current - margin
            win_xright_high = rightx_current + margin

            # Identify the nonzero pixels in x and y within the window #
            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                              (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                               (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]

            # Append these indices to the lists
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)

            # If you found > minpix pixels, recenter next window on their mean position
            if len(good_left_inds) > minpix:
                leftx_current = int(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > minpix:
                rightx_current = int(np.mean(nonzerox[good_right_inds]))

        # Concatenate the arrays of indices (previously was a list of lists of pixels)
        try:
            left_lane_inds = np.concatenate(left_lane_inds)
            right_lane_inds = np.concatenate(right_lane_inds)
        except ValueError:
            # Avoids an error if the above is not implemented fully
            pass

        # Extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]

        return leftx, lefty, rightx, righty

    def fit_poly(self, binary_warped, leftx, lefty, rightx, righty):
        ### Fit a second order polynomial to each with np.polyfit() ###
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)

        # Generate x and y values for plotting
        ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])
        try:
            left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
            right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]
        except TypeError:
            # Avoids an error if `left` and `right_fit` are still none or incorrect
            print('The function failed to fit a line!')
            left_fitx = 1 * ploty ** 2 + 1 * ploty
            right_fitx = 1 * ploty ** 2 + 1 * ploty

        return left_fit, right_fit, left_fitx, right_fitx, ploty

    def draw_poly_lines(self, binary_warped, left_fitx, right_fitx, ploty):
        # Create an image to draw on and an image to show the selection window
        out_img = np.dstack((binary_warped, binary_warped, binary_warped))
        window_img = np.zeros_like(out_img)

        margin = 10
        # Generate a polygon to illustrate the search window area
        # And recast the x and y points into usable format for cv2.fillPoly()
        left_line_window1 = np.array([np.transpose(np.vstack([left_fitx - margin, ploty]))])
        left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx + margin,
                                                                        ploty])))])
        left_line_pts = np.hstack((left_line_window1, left_line_window2))
        right_line_window1 = np.array([np.transpose(np.vstack([right_fitx - margin, ploty]))])
        right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx + margin,
                                                                         ploty])))])
        right_line_pts = np.hstack((right_line_window1, right_line_window2))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(window_img, np.int_([left_line_pts]), (255, 255, 0))
        cv2.fillPoly(window_img, np.int_([right_line_pts]), (255, 255, 0))
        result = cv2.addWeighted(out_img, 1, window_img, 0.5, 0)
        cv2.imshow("lane", result)
        return window_img

    def find_lane_pixels_using_prev_poly(self, binary_warped):
        # width of the margin around the previous polynomial to search
        margin = 50
        # Grab activated pixels
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        ### Set the area of search based on activated x-values ###
        ### within the +/- margin of our polynomial function ###
        left_lane_inds = ((nonzerox > (self.prev_left_fit[0] * (nonzeroy ** 2) + self.prev_left_fit[1] * nonzeroy + self.prev_left_fit[2] - margin))
                          & (nonzerox < (self.prev_left_fit[0] * (nonzeroy ** 2) + self.prev_left_fit[1] * nonzeroy + self.prev_left_fit[2] + margin))).nonzero()[0]
        right_lane_inds = ((nonzerox > (self.prev_right_fit[0] * (nonzeroy ** 2) + self.prev_right_fit[1] * nonzeroy + self.prev_right_fit[2] - margin))
                          & (nonzerox < (self.prev_right_fit[0] * (nonzeroy ** 2) +  self.prev_right_fit[1] * nonzeroy + self.prev_right_fit[2] + margin))).nonzero()[0]

        # Again, extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]

        return leftx, lefty, rightx, righty

    def lane_finding_pipeline(self, binary_warped):
        if (len(self.left_fit_hist) == 0):
            leftx, lefty, rightx, righty = self.find_lane_pixels_using_histogram(binary_warped)
            left_fit, right_fit, left_fitx, right_fitx, ploty = self.fit_poly(binary_warped, leftx, lefty, rightx, righty)
            # Store fit in history
            self.left_fit_hist = np.array(left_fit)
            new_left_fit = np.array(left_fit)
            self.left_fit_hist = np.vstack([self.left_fit_hist, new_left_fit])
            self.right_fit_hist = np.array(right_fit)
            new_right_fit = np.array(right_fit)
            self.right_fit_hist = np.vstack([self.right_fit_hist, new_right_fit])
        else:
            self.prev_left_fit = [np.mean(self.left_fit_hist[:, 0]), np.mean(self.left_fit_hist[:, 1]), np.mean(self.left_fit_hist[:, 2])]
            self.prev_right_fit = [np.mean(self.right_fit_hist[:, 0]), np.mean(self.right_fit_hist[:, 1]), np.mean(self.right_fit_hist[:, 2])]
            leftx, lefty, rightx, righty = self.find_lane_pixels_using_prev_poly(binary_warped)
            if (len(lefty) == 0 or len(righty) == 0):
                leftx, lefty, rightx, righty = self.find_lane_pixels_using_histogram(binary_warped)
            left_fit, right_fit, left_fitx, right_fitx, ploty = self.fit_poly(binary_warped, leftx, lefty, rightx, righty)

            # Add new values to history
            new_left_fit = np.array(left_fit)
            self.left_fit_hist = np.vstack([self.left_fit_hist, new_left_fit])
            new_right_fit = np.array(right_fit)
            self.right_fit_hist = np.vstack([self.right_fit_hist, new_right_fit])

            # Remove old values from history
            if (len(self.left_fit_hist) > 10):
                self.left_fit_hist = np.delete(self.left_fit_hist, 0, 0)
                self.right_fit_hist = np.delete(self.right_fit_hist, 0, 0)

        return left_fitx, right_fitx, ploty, left_fit, right_fit

    def measure_curvature_meters(self, img, left_fitx, right_fitx, ploty):
        # Define conversions in x and y from pixels space to meters
        ym_per_pix = 30 / img.shape[0]  # meters per pixel in y dimension
        xm_per_pix = 3.7 / img.shape[1]  # meters per pixel in x dimension

        left_fit_cr = np.polyfit(ploty * ym_per_pix, left_fitx * xm_per_pix, 2)
        right_fit_cr = np.polyfit(ploty * ym_per_pix, right_fitx * xm_per_pix, 2)
        # Define y-value where we want radius of curvature
        # We'll choose the maximum y-value, corresponding to the bottom of the image
        y_eval = np.max(ploty)

        # Calculation of R_curve (radius of curvature)
        left_curverad = ((1 + (2 * left_fit_cr[0] * y_eval * ym_per_pix + left_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
            2 * left_fit_cr[0])
        right_curverad = ((1 + (
                    2 * right_fit_cr[0] * y_eval * ym_per_pix + right_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
            2 * right_fit_cr[0])

        return left_curverad, right_curverad

    def road_info(self, img, left_fitx, right_fitx, ploty):
        """Road info"""
        # curvature
        left_curverad, right_curverad = self.measure_curvature_meters(img, left_fitx, right_fitx, ploty)
        radius_of_curvature = (left_curverad + right_curverad) / 2
        curvature = {
            'left_curvature': left_curverad, 'right_curvature': right_curverad}
        # print(f"Curvature: {curvature}")

        left_startx, right_startx = left_fitx[len(left_fitx) - 1], right_fitx[len(right_fitx) - 1]  # bottom component
        left_endx, right_endx = left_fitx[0], right_fitx[0]  # top components

        direction = ((left_endx - left_startx) + (right_endx - right_startx)) / 2
        # print("direction :", direction)

        if radius_of_curvature > 2000 and abs(direction) < 100:
            road_info = 'No Curve'
            curvature = -1
        elif radius_of_curvature <= 2000 and direction < -50:
            road_info = 'Left Curve'
        elif radius_of_curvature <= 2000 and direction > 50:
            road_info = 'Right Curve'
        else:
            road_info = 'Undetectable'

        # deviation
        center_lane = (right_startx + left_startx) / 2
        lane_width = right_startx - left_startx

        center_car = 720 / 2 + 12

        pix_deviation = round((center_lane - center_car) / (lane_width / 2), 6)
        deviation = round((3.5) * pix_deviation / lane_width, 2)

        if center_lane > center_car:
            deviation_state = f'Left {str(abs(deviation))}m'
        elif center_lane < center_car:
            deviation_state = f'Right {str(abs(deviation))}m'
        else:
            deviation_state = 'Center'

        # steering

        if radius_of_curvature > 2000:
            steering_angle = 0
        else:
            wheelbase = 2700  # mm
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
        return road_info

    def forward(self, img):
        # img[:100, :] = 0
        img[:,550:] = 0
        left_fitx, right_fitx, ploty, left_fit, right_fit = self.lane_finding_pipeline(img)
        road_info = self.road_info(img, left_fitx, right_fitx, ploty)
        img = self.draw_poly_lines(img, left_fitx, right_fitx, ploty)
        return img, road_info, left_fitx, right_fitx, ploty, left_fit, right_fit

