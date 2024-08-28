import cv2
import numpy as np

class Preprocessing:
    def __init__(self):
        """ Init Thresholding """
        pass

    # 영상에서 프레임을 추출하여 사용
    def undistort_const(self, frame):
        
        mtx = np.array([[2000, 0, 320],
                        [0, 2000, 180],
                        [0, 0, 1]], dtype=np.float32)
                
        # 왜곡 계수 설정 (임의의 값 또는 왜곡이 없다고 가정)
        # dist = np.zeros((5, 1))
        dist = np.array([-0.5, -0.5, 0.0, 0.0, 0.0])

        # 프레임 보정
        h, w = frame.shape[:2]
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
        undistorted_frame = cv2.undistort(frame, mtx, dist, None, newcameramtx)

        #outline
        cropped = undistorted_frame[5:h-5, 5:w-5]

        # 이미지를 원래 크기로 확대
        resized_frame = cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)
        
        return resized_frame