"""
ADAS projects by Autopilot

Functions :
    Lane Detection
    Object Detection by Yolo
    ...
"""
import cv2

from new_LaneLines import *
from Yolo_v8 import *
from Preprocessing import *
from Notice import *
from BrakeDetector import *
from LaneChangeDetector_yolov8 import *
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
        self.model = self.yolo.model
        self.notice = Notice()
        self.tail = KalmanBrakeDetector()
        self.lane_change_detector = LaneChangeDetector()

        self.lane_change = 2
        self.cnt = 0
        self.braking = False

    def remove_car_boxes(self, img, boxes):
        # results = self.model.predict(img, device=device)
        # backward_img = results[0].plot()
        # cv2.namedWindow("backward_img")
        # cv2.moveWindow("backward_img", 500, 0)
        # cv2.imshow("backward_img", backward_img)

        # boxes = results[0].boxes  # Yolov8의 예측 결과 상자들
        if len(boxes) > 0:
            boxes_array = boxes.xyxy.cpu().numpy()
            classes = boxes.cls.cpu().numpy()

            for box, cls in zip(boxes_array, classes):
                if self.model.names[int(cls)] == 'car':
                    x1, y1, x2, y2 = map(int, box)  # 바운딩 박스 좌표를 정수로 변환
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 0), -1)  # 검은색 박스로 지우기

        return img

    def forward(self, img, boxes):
        out_img = np.copy(img)
        out_img2 = np.copy(img)
        # print(f"out_img: {out_img}")
        # cv2.imshow("out_img",out_img)
        time1 = time.perf_counter_ns()

        img_rm_car = self.remove_car_boxes(out_img2, boxes)
        img_rm_car = np.where(img_rm_car != 0, 255, img_rm_car).astype(np.uint8)
        gray_rm_car = cv2.cvtColor(img_rm_car, cv2.COLOR_BGR2GRAY)
        img = self.thresholding.forward(img)
        # img = cv2.bitwise_and(gray_rm_car, img)
        # cv2.imshow("img2", img)
        time2 = time.perf_counter_ns()
        # cv2.imshow("self.transform.forward(img)", img)
        img = self.transform.forward(img)
        M_inv = self.transform.M_inv
        # cv2.imshow("self.thresholding.forward(img)", img)
        time3 = time.perf_counter_ns()
        img, road_info, left_line, right_line, y, left_fit, right_fit = self.lanelines.forward(img)
        time4 = time.perf_counter_ns()
        img = self.transform.backward(img)
        time5 = time.perf_counter_ns()
        out_img = cv2.addWeighted(out_img, 1, img, 0.6, 0)
        # print(f"time1: {check_time(time1, time2)}ms, time2: {check_time(time2, time3)}ms, time3: {check_time(time3, time4)}ms, time4: {check_time(time4, time5)}ms")
        return img, road_info, left_line, right_line, M_inv, y, left_fit, right_fit

    def process_image(self, img_path):
        cap = cv2.VideoCapture(img_path)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 코덱 설정 (예: 'mp4v', 'XVID', 'MJPG')
        frame_width, frame_height = 640, 360  # 리사이즈된 해상도
        out = cv2.VideoWriter('test_11_results.mp4', fourcc, 20.0, (frame_width, frame_height))
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

            yolo_results = self.model(frame)
            # 탐지된 객체들에 대한 정보를 가져옴
            boxes = yolo_results[0].boxes

            lane_img = np.copy(frame)
            time1 = time.perf_counter_ns()
            """ -------Yolo process------- """
            yolo_img, distance, front_car_boxes, distance_text = self.yolo.detect_and_calculate_distance(frame, boxes)
            """ --------------------------- """
            time2 = time.perf_counter_ns()


            # current_time = time.time()
            # if current_time - last_brake_analysis_time >= brake_analysis_interval:

            if front_car_boxes is not None:
                self.braking = self.tail.forward(frame, front_car_boxes)
                # cv2.imshow("tail_img", tail_img)

            # last_brake_analysis_time = current_time

            # elapsed_time = time.time() - start_time_1
            # if elapsed_time < video_frame_interval:
            #     time.sleep(video_frame_interval - elapsed_time)

            lane_img, road_info, left_line, right_line, M_inv, y, left_fit, right_fit = self.forward(lane_img, boxes)
            if left_line is not None:
                lane_change = self.lane_change_detector.detect_lane_change(front_car_boxes, left_line, right_line, M_inv, y, left_fit, right_fit)
                self.lane_change = lane_change
                # 차선 변경 여부 텍스트 출력
                if lane_change == 0:
                    cv2.putText(yolo_img, "Changing to left Lane", (50, 50), cv2.FONT_HERSHEY_SIMPLEX,
                                1, (0, 0, 255), 2, cv2.LINE_AA)
                elif lane_change == 1:
                    cv2.putText(yolo_img, "Changing to right Lane", (50, 50), cv2.FONT_HERSHEY_SIMPLEX,
                                1, (0, 255, 0), 2, cv2.LINE_AA)
                else:
                    pass


            time3 = time.perf_counter_ns()
            result = cv2.addWeighted(lane_img, 0.5, yolo_img, 1, 0)
            # 프레임 종료 시간 기록
            end_time = time.perf_counter_ns()
            # 프레임 당 소요 시간 계산
            frame_time_ns = end_time - start_time
            frame_time_ms = frame_time_ns / 1_000_000
            # 프레임 시간 출력
            # print(f"Frame time: {frame_time_ms:.2f} ms")

            #UI
            # 핸들 회전 시각화
            if road_info is not None:
                result = self.notice.combine(result, road_info[3])

            # 우선 순위: 차간 거리 > 차선 이탈 > 차선 변경
            # 차간 거리 경고
            if self.braking:
                result = self.notice.sign(result)

            if distance < 20:
                result = self.notice.red_sign(result)
            else:
                # 차선 이탈 경고
                if road_info is not None:
                    if self.lane_change != 2:
                        result = self.notice.blue_sign(result, self.lane_change)

                    # 차선 변경 경고
                    elif abs(road_info[4]) > 0.0026:
                        result = self.notice.green_sign(result, road_info[4])
                        self.cnt += 1
                    else:
                        self.cnt = 0

            result = self.notice.shadow(result)
            cv2.putText(result, f"Frame time: {frame_time_ms:.2f} ms", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (255, 255, 255), 1, lineType=cv2.LINE_AA)
            cv2.putText(result, distance_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, lineType=cv2.LINE_AA)
            cv2.imshow("result", result)
            print(f"time1: {check_time(time1, time2)}ms, time2: {check_time(time2, time3)}ms")
            out.write(result)

            if cv2.waitKey(10) == 27:
                break



def main():
    # img_path = "test_7.mp4" # 조향, 차선유지 등 전반적인 기능
    # img_path = "test_7_2.mp4" # 후미등 점멸하는 영상
    # img_path = "test_8.mp4" # 후미등으로 앞 차선 변경하는 정도만
    img_path = "test_11.mp4" # 차선책

    findLaneLines = FindLaneLines()
    findLaneLines.process_image(img_path)


if __name__ == "__main__":
    main()