"""
ADAS projects by Autopilot

Functions :
    Lane Detection
    Object Detection by Yolo
    ...
"""

import numpy as np
import cv2
from Preprocessing import *
from Thresholding import *
from PerspectiveTransformation import *
from LaneLines import *
from Yolo import *
from Notice import *
import time

class FindLaneLines :
    def __init__(self):
        """ Init Application """
        self.thresholding = Thresholding()
        self.transform = PerspectiveTransformation()
        self.lanelines = LaneLines()
        self.notice = Notice()

    def forward(self, img):
        out_img = np.copy(img)
        img = self.transform.forward(img)
        cv2.imshow("self.transform.forward(img)", self.transform.forward(img))
        # print(f"self.transform.forward(img): {self.transform.forward(img)}")
        img = self.thresholding.forward(img)
        # print(f"self.thresholding.forward(img):{self.thresholding.forward(img).shape}")
        img, road_info = self.lanelines.forward(img)
        img = self.transform.backward(img)

        out_img = cv2.addWeighted(out_img, 1, img, 0.6, 0)

        return out_img, road_info
    

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
            frame = cv2.resize(frame, (640, 360), interpolation=cv2.INTER_LINEAR) 

            if not ret:
                print("--------Video Ended---------")
                break

            #callibration & resize
            undistort_frame = Preprocessing.undistort_const(frame)

            #lane detective
            lane_img, road_info = self.forward(undistort_frame)

            """ -------Yolo process------ - """
            # ROI 처리
            roi_img = yolo.normalize_ROI(undistort_frame)

            # YOLO 객체 검출
            boxes, confidences, class_ids = yolo.object_YOLO(roi_img)

            if boxes:
                largest_box = yolo.get_largest_object(boxes, confidences, class_ids)

                yolo_img, distance = yolo.draw_largest_box(undistort_frame, largest_box)
                distance = str(round(distance,1))
                
                result = cv2.addWeighted(lane_img, 0.5, yolo_img, 0.5, 0)
            else:
                distance = "No vehicles"
                result = lane_img

            # 가장 큰 박스 검출
            largest_box = yolo.get_largest_object(boxes, confidences, class_ids)

            # 바운딩 박스 그리기
            #yolo_img = yolo.draw_bounding_boxes(frame, boxes, confidences, class_ids)
            """ --------------------------- """


            cv2.putText(result, f"distance: {distance}", (10, 55), cv2.FONT_HERSHEY_COMPLEX, 0.7, (100, 100, 200), 1)

            cv2.putText(result, f"road_info: {road_info[1]}", (10, 80), cv2.FONT_HERSHEY_COMPLEX, 0.45, (100, 100, 200), 1)
            cv2.putText(result, f"deviation: {road_info[2]}", (10, 100), cv2.FONT_HERSHEY_COMPLEX, 0.45, (100, 100, 200), 1)

            result = self.notice.combine(result, road_info[3])

            cv2.imshow('result', result)

            if cv2.waitKey(10) == 27:
                break

def main():
    img_path = "input.mp4"

    findLaneLines = FindLaneLines()
    findLaneLines.process_image(img_path)

if __name__ == "__main__":
    main()