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
from Yolo import *
import time

class FindLaneLines :
    def __init__(self):
        """ Init Application """
        self.thresholding = Thresholding()
        self.transform = PerspectiveTransformation()
        self.lanelines = LaneLines()

    def forward(self, img):
        out_img = np.copy(img)
        # print(f"out_img: {out_img}")
        cv2.imshow("out_img",out_img)
        img = self.transform.forward(img)
        cv2.imshow("self.transform.forward(img)", self.transform.forward(img))
        # print(f"self.transform.forward(img): {self.transform.forward(img)}")
        img = self.thresholding.forward(img)
        # print(f"self.thresholding.forward(img):{self.thresholding.forward(img).shape}")
        img = self.lanelines.forward(img)
        img = self.transform.backward(img)

        out_img = cv2.addWeighted(out_img, 1, img, 0.6, 0)

        return out_img

    def process_image(self, img_path):
        cap = cv2.VideoCapture(img_path)

        """ -------Yolo Init------- """
        yolo = YOLO('yolov3.weights', 'yolov3.cfg', 'coco.names')
        """ ----------------------- """

        if not cap.isOpened():
            print(f"Error: Failed to open video from {'sample1.avi'}")
            exit(1)

        while cap.isOpened():
            ret, frame = cap.read()
            # 프레임 시작 시간 기록
            start_time = time.perf_counter_ns()

            if not ret:
                print("--------Video Ended---------")
                break


            """ -------Yolo process------- """
            # ROI 처리
            roi_img = yolo.normalize_ROI(frame)

            # YOLO 객체 검출
            boxes, confidences, class_ids = yolo.object_YOLO(roi_img)

            # 바운딩 박스 그리기
            yolo_img = yolo.draw_bounding_boxes(frame, boxes, confidences, class_ids)

            """ --------------------------- """

            lane_img = self.forward(frame)

            result = cv2.addWeighted(lane_img, 0.5, yolo_img, 0.5, 0)
            # 프레임 종료 시간 기록
            end_time = time.perf_counter_ns()
            # 프레임 당 소요 시간 계산
            frame_time_ns = end_time - start_time
            frame_time_ms = frame_time_ns / 1_000_000
            # 프레임 시간 출력
            # print(f"Frame time: {frame_time_ms:.2f} ms")
            cv2.putText(result, f"Frame time: {frame_time_ms:.2f} ms", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (255, 255, 255), 2)
            cv2.imshow("result", result)

            if cv2.waitKey(10) == 27:
                break

def main():
    img_path = "sample1.avi"

    findLaneLines = FindLaneLines()
    findLaneLines.process_image(img_path)

if __name__ == "__main__":
    main()