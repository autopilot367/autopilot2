"""
ADAS projects by Autopilot

Functions :
    Lane Detection
    Object Detection by Yolo
    ...
"""

import numpy as np
import cv2
from Thresholding import *
from PerspectiveTransformation import *
from LaneLines import *
from Yolo_v8 import *
from Preprocessing import *
from Notice import *
from BrakeDetector import *
import time
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


# 전체 루프 FPS 설정 (비디오의 FPS와 동일하게 설정)



def check_time(start_time, end_time):
    """
        time = time.perf_counter_ns()
    """
    frame_time_ns = end_time - start_time
    frame_time_ms = frame_time_ns / 1_000_000
    return frame_time_ms

class FindLaneLines :
    def __init__(self):
        """ Init Application """
        self.thresholding = Thresholding()
        self.transform = PerspectiveTransformation()
        self.lanelines = LaneLines()
        self.calibration = Preprocessing()
        self.yolo = YOLOv8CarDetector('yolov8n.pt')
        self.notice = Notice()
        self.tail = KalmanBrakeDetector()



    def forward(self, img):
        out_img = np.copy(img)
        # print(f"out_img: {out_img}")
        cv2.imshow("out_img",out_img)
        time1 = time.perf_counter_ns()
        img = self.transform.forward(img)
        cv2.imshow("self.transform.forward(img)", img)
        # print(f"self.transform.forward(img): {self.transform.forward(img)}")
        time2 = time.perf_counter_ns()
        img = self.thresholding.forward(img)
        # print(f"self.thresholding.forward(img):{self.thresholding.forward(img).shape}")
        time3 = time.perf_counter_ns()
        img, road_info = self.lanelines.forward(img)
        time4 = time.perf_counter_ns()
        img = self.transform.backward(img)
        time5 = time.perf_counter_ns()
        out_img = cv2.addWeighted(out_img, 1, img, 0.6, 0)
        # print(f"time1: {check_time(time1, time2)}ms, time2: {check_time(time2, time3)}ms, time3: {check_time(time3, time4)}ms, time4: {check_time(time4, time5)}ms")
        return out_img, road_info

    def process_image(self, img_path):
        cap = cv2.VideoCapture(img_path)
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        video_frame_interval = 1 / video_fps

        # 브레이크등 분석을 위한 FPS 설정 (예: 0.2초 간격으로 분석)
        brake_analysis_interval = 0.2  # 초 단위로 간격 설정
        # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)

        if not cap.isOpened():
            print(f"Error: Failed to open video from {'sample1.avi'}")
            exit(1)

        last_brake_analysis_time = time.time()

        while cap.isOpened():
            ret, frame = cap.read()
            # 프레임 시작 시간 기록
            if not ret:
                print("--------Video Ended---------")
                break
            frame = cv2.resize(frame, (640, 360), interpolation=cv2.INTER_LINEAR)
            start_time_1 = time.time()
            start_time = time.perf_counter_ns()


            frame = self.calibration.undistort_const(frame)


            lane_img = np.copy(frame)
            time1 = time.perf_counter_ns()
            """ -------Yolo process------- """
            yolo_img, distance, front_car_boxes = self.yolo.detect_and_calculate_distance(frame)
            """ --------------------------- """
            time2 = time.perf_counter_ns()


            current_time = time.time()
            if current_time - last_brake_analysis_time >= brake_analysis_interval:

                if front_car_boxes is not None:
                    frame = self.tail.forward(frame, front_car_boxes)
                    # cv2.imshow("tail_img", tail_img)

                last_brake_analysis_time = current_time

            elapsed_time = time.time() - start_time_1
            if elapsed_time < video_frame_interval:
                time.sleep(video_frame_interval - elapsed_time)

            lane_img, road_info = self.forward(lane_img)
            time3 = time.perf_counter_ns()
            result = cv2.addWeighted(lane_img, 0.5, yolo_img, 0.5, 0)
            # 프레임 종료 시간 기록
            end_time = time.perf_counter_ns()
            # 프레임 당 소요 시간 계산
            frame_time_ns = end_time - start_time
            frame_time_ms = frame_time_ns / 1_000_000
            # 프레임 시간 출력
            # print(f"Frame time: {frame_time_ms:.2f} ms")

            if road_info is not None:
                # 핸들 회전 시각화
                result = self.notice.combine(result, road_info[3])
            # 차간 거리 경고
            if distance:
                if distance < 10:  # 임의로 설정한 값, 탑승 중인 차량의 속력 추가 필요
                    result = self.notice.red_sign(result)

            cv2.putText(result, f"Frame time: {frame_time_ms:.2f} ms", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (255, 255, 255), 2)

            cv2.imshow("result", result)
            print(f"time1: {check_time(time1, time2)}ms, time2: {check_time(time2, time3)}ms")
            if cv2.waitKey(10) == 27:
                break

def main():
    img_path = "test_1.mp4"

    findLaneLines = FindLaneLines()
    findLaneLines.process_image(img_path)

if __name__ == "__main__":
    main()