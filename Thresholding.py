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

    def forward(self, frame):
        # """ Take an image and extract all relevant pixels.
        #
        # Parameters:
        #     img (np.array) : Input image
        #
        # Returns :
        #     binary (np.array) : A binary image represent all positions of relevant pixels.
        # """

        img = frame
        img = cv2.GaussianBlur(img, (5, 5), 1)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        """test_7.mp4"""
        h, s, v = cv2.split(hsv)
        v_gaussian = cv2.adaptiveThreshold(v, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 127, -21) # test_7
        # mask = v_gaussian
        # return mask

        lower_white_optimal = np.array([0, 0, 180])
        upper_white_optimal = np.array([180, 80, 255])

        lower_yellow_optimal = np.array([15, 70, 100])
        upper_yellow_optimal = np.array([40, 255, 255])

        lower_pink = np.array([100, 15, 15])  # 분홍색 시작 범위 (더 확장)
        upper_pink = np.array([180, 255, 255])  # 분홍색 끝 범위

        # Optimal HSV ranges for white and yellow lanes
        mask_white = cv2.inRange(hsv, lower_white_optimal, upper_white_optimal)
        mask_yellow = cv2.inRange(hsv, lower_yellow_optimal, upper_yellow_optimal)

        mask_pink = cv2.inRange(hsv, lower_pink, upper_pink)

        # 분홍색 영역을 제외한 마스크 생성
        mask_no_pink = cv2.bitwise_not(mask_pink)

        # 흰색과 노란색 차선 마스크에 분홍색 영역 제외
        mask_lanes = cv2.bitwise_and(cv2.bitwise_or(mask_white, mask_yellow), mask_no_pink)
        mask_lanes = cv2.bitwise_and(mask_lanes, v_gaussian)
        return mask_lanes


        # # 최적 파라미터 설정
        # sobel_thresh_min = 50  # Sobel 필터의 하한 임계값을 더 높여 노이즈 감소
        # sobel_thresh_max = 255
        # white_thresh_min = 210  # 더 밝은 흰색만 검출하도록 임계값 조정
        # white_thresh_max = 255
        # hue_thresh_min = 20  # 노란색에 더 집중하기 위해 Hue 범위를 더욱 좁힘
        # hue_thresh_max = 30
        # sat_thresh_min = 150  # 채도 값의 하한선을 더 높여 불필요한 영역 필터링
        # sat_thresh_max = 255
        #
        # kernel = np.ones((7, 7), np.uint8)  # 커널 사이즈를 9x9로 키워 노이즈 제거 강화
        #
        # # Transform image to gray scale
        # gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        #
        # # Sobel 필터 적용
        # sobelx = cv2.Sobel(gray_frame, cv2.CV_64F, 1, 0, ksize=15)
        # abs_sobelx = np.absolute(sobelx)
        # scaled_sobel = np.uint8(255 * abs_sobelx / np.max(abs_sobelx))
        # sx_binary = np.zeros_like(scaled_sobel)
        # sx_binary[(scaled_sobel >= sobel_thresh_min) & (scaled_sobel <= sobel_thresh_max)] = 1
        #
        # # 흰색 차선 감지
        # white_binary = np.zeros_like(gray_frame)
        # white_binary[(gray_frame > white_thresh_min) & (gray_frame <= white_thresh_max)] = 1
        #
        # # HLS 색상 공간으로 변환 및 노란색 차선 감지
        # hls_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HLS)
        # H = hls_frame[:, :, 0]
        # S = hls_frame[:, :, 2]
        # sat_binary = np.zeros_like(S)
        # sat_binary[(S > sat_thresh_min) & (S <= sat_thresh_max)] = 1
        # hue_binary = np.zeros_like(H)
        # hue_binary[(H > hue_thresh_min) & (H <= hue_thresh_max)] = 1
        #
        # # 감지된 모든 요소를 결합
        # binary_1 = cv2.bitwise_or(sx_binary, white_binary)
        # binary_2 = cv2.bitwise_or(hue_binary, sat_binary)
        # binary = cv2.bitwise_or(binary_1, binary_2)
        # binary_morphed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        # binary_display = binary_morphed * 255
        # cv2.imshow("binary", binary_display)
        # return binary_display
