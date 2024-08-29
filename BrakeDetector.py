import cv2
import time
import numpy as np
from Thresholding import *
from PerspectiveTransformation import *
from LaneLines import *
from yolo_test import *
from ADAS_main import FindLaneLines
import CarBehaviour as cb

class KalmanBrakeDetector:
    def __init__(self):
        # Kalman Filter 초기화
        self.kalman = cv2.KalmanFilter(4, 2)
        self.kalman.measurementMatrix = np.array([[1, 0, 0, 0], 
                                                  [0, 1, 0, 0]], np.float32)
        self.kalman.transitionMatrix = np.array([[1, 0, 1, 0], 
                                                 [0, 1, 0, 1], 
                                                 [0, 0, 1, 0], 
                                                 [0, 0, 0, 1]], np.float32)
        self.kalman.processNoiseCov = np.array([[1, 0, 0, 0],
                                                [0, 1, 0, 0],
                                                [0, 0, 1, 0],
                                                [0, 0, 0, 1]], np.float32) * 0.03
        self.brake_status = 0

        self.prev_frame = None
        self.current_frame = None

    def update(self, brake_light_detected):
        # Kalman Filter에 현재 관측값을 전달
        if isinstance(brake_light_detected, (int, float)):  # brake_light_detected가 단일 값인 경우
            measurement = np.array([[np.float32(brake_light_detected)],
                                    [0]], np.float32)
        else:
            raise ValueError("brake_light_detected는 단일 값이어야 합니다.")
        self.kalman.correct(measurement)
        
        # 필터를 통해 예측된 상태 업데이트
        prediction = self.kalman.predict()
        predicted_brake_status = prediction[0][0]

        # 예측된 상태를 기반으로 braking 여부 결정
        self.brake_status = 1 if predicted_brake_status > 0.5 else 0

        return self.brake_status

    def roi_for_tail_detect(self, frame, front_car_boxes):
        x1, y1, w, h = front_car_boxes

        # ROI 선정 (tail 부분으로 특정 부분만 크롭)
        roi_frame = frame[y1 + int(0.3 * h):y1 + int(0.8 * h), x1 + int(0.08 * w):x1 + int(0.92 * w)]

        return roi_frame

    def detect_status_change(self, roi_frame):
        self.current_frame = roi_frame
        status = 0
        if self.prev_frame is not None:
            difference, non1, non2 = cb.CarBehaviour(self.prev_frame, self.current_frame)
            if difference is not None:
                print('difference :' , difference / cb.count_pixels(self.prev_frame), 'one only :' , non1 / cb.count_pixels(self.prev_frame))
                if (difference / cb.count_pixels(self.prev_frame) <= -0.001) or  (non1 / cb.count_pixels(self.prev_frame) >= 0.005) : status = True
                else : status = False

        self.prev_frame = roi_frame
        braking = self.update(status)
        return braking

    def forward(self, frame, front_car_boxes):
        roi_frame = self.roi_for_tail_detect(frame, front_car_boxes)
        braking = self.detect_status_change(roi_frame)
        if braking == True:
            cv2.putText(frame, "Braking", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        return frame
# # TailDetect_main에서 KalmanBrakeDetector를 사용하는 코드
# kalman_brake_detector = KalmanBrakeDetector()
# # 동영상 파일 열기
# cap = cv2.VideoCapture('road_10.mp4')
#
# # 전체 루프 FPS 설정 (비디오의 FPS와 동일하게 설정)
# video_fps = cap.get(cv2.CAP_PROP_FPS)
# video_frame_interval = 1 / video_fps
#
# # 브레이크등 분석을 위한 FPS 설정 (예: 0.2초 간격으로 분석)
# brake_analysis_interval = 0.2  # 초 단위로 간격 설정
#
# # 첫 번째 프레임 읽기
# ret, prev_frame = cap.read()
#
# if ret :
#     prev_frame = roi_for_tail_detect(prev_frame)
#
# # 브레이크등 분석을 위한 타이머 초기화
# last_brake_analysis_time = time.time()
#
# while ret:
#     start_time = time.time()
#     status = []
#     # 다음 프레임 읽기
#     ret, frame = cap.read()
#     if not ret:
#         break
#
#     current_frame = roi_for_tail_detect(frame)
#     # #전체 루프 내에서 차선 및 전방 차량 감지 수행
#     # findLaneLines = FindLaneLines()
#     # lane_img = findLaneLines.forward(current_frame)
#
#     # 브레이크등 분석 간격 확인
#     current_time = time.time()
#     if current_time - last_brake_analysis_time >= brake_analysis_interval:
#
#         # 이전 프레임과 비교하여 브레이크등 상태 변화 감지
#         if prev_frame is not None:
#             difference, non1, non2 = cb.CarBehaviour(prev_frame, current_frame)
#             print('difference :' , difference / cb.count_pixels(prev_frame), 'one only :' , non1 / cb.count_pixels(prev_frame))
#             if (difference / cb.count_pixels(prev_frame) <= -0.001) or  (non1 / cb.count_pixels(prev_frame) >= 0.01) : status = True
#             else : status = False
#
#         braking = kalman_brake_detector.update(status)
#
#         if braking == True :
#             cv2.putText(frame, "Braking", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
#
#         # 분석 타이머 갱신 및 이전 프레임 업데이트
#         last_brake_analysis_time = current_time
#         prev_frame = current_frame.copy()
#
#     # 결과를 화면에 표시
#     cv2.imshow('Frame', frame)
#
#     # 루프 실행 시간을 고려하여 전체 루프 FPS를 유지
#     elapsed_time = time.time() - start_time
#     if elapsed_time < video_frame_interval:
#         time.sleep(video_frame_interval - elapsed_time)
#
#     #ESC키를 누르면 루프 종료
#     if cv2.waitKey(10) == 27:
#         break
#
#     # result = cv2.addWeighted(lane_img, 0.5, current_frame, 0.5, 0)
#     # # 프레임 종료 시간 기록
#     # end_time = time.perf_counter_ns()
#     # # 프레임 당 소요 시간 계산
#     # frame_time_ns = end_time - start_time
#     # frame_time_ms = frame_time_ns / 1_000_000
#     # # 프레임 시간 출력
#     # # print(f"Frame time: {frame_time_ms:.2f} ms")
#     # cv2.putText(result, f"Frame time: {frame_time_ms:.2f} ms", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
#     #                     (255, 255, 255), 2)
#     # cv2.imshow("result", result)
#
#     # if cv2.waitKey(10) == 27:
#     #     break
#
# # 모든 작업이 끝난 후, 리소스 해제
# cap.release()
# cv2.destroyAllWindows()
