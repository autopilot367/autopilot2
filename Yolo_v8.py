import cv2
from ultralytics import YOLO
import torch
import sys

class YOLOv8CarDetector:
    def __init__(self, model_path, conf_threshold=0.5):
        # YOLOv8 모델 로드 및 GPU 사용 설정
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold

        # CUDA 사용 설정
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        print(f"모델이 {self.device}에서 실행됩니다.")

    # def detect_and_draw(self, frame):
    #     # YOLOv8 모델을 사용하여 프레임에서 객체 탐지
    #     results = self.model(frame)
    #
    #     # 결과 프레임에 바운딩 박스 및 라벨을 그리기
    #     annotated_frame = results[0].plot()
    #
    #     return annotated_frame

    def detect_and_calculate_distance(self, frame, boxes, focal_length=1000, real_car_width=2.0):
        # YOLOv8 모델을 사용하여 프레임에서 객체 탐지
        # results = self.model(frame)

        # 탐지된 객체들에 대한 정보를 가져옴
        # boxes = results[0].boxes

        if len(boxes) == 0:
            return frame, None  # 탐지된 객체가 없을 경우 원본 프레임 반환

        # 가장 큰 박스를 찾기 위한 변수 초기화
        max_area = 0
        largest_vehicle_box = None
        largest_vehicle_class_id = None
        # 프레임의 중심 X 좌표 및 탐지 범위 설정
        frame_center_x = frame.shape[1] / 2
        detection_range_min = frame_center_x - frame.shape[1] * 0.1
        detection_range_max = frame_center_x + frame.shape[1] * 0.1

        for box in boxes:
            # 바운딩 박스 좌표 (x1, y1, x2, y2)
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            class_id = int(box.cls[0])  # 클래스 ID

            # 차량으로 인식된 객체만 고려 (예: 클래스 ID 2 = 차량)
            if class_id in [2, 3, 5, 7]:  # 차량, 트럭, 버스 등 (필요에 따라 클래스 ID 수정)
                # 박스의 넓이와 높이 계산
                width = x2 - x1
                height = y2 - y1

                # 박스의 면적 계산
                area = width * height

                # 중심 X 좌표 계산
                centroid_x = (x1 + x2) / 2

                # 프레임 중앙의 일정 범위 내에 있는 객체만 고려
                if detection_range_min <= centroid_x <= detection_range_max:
                    # 최대 면적과 비교하여 가장 큰 차량을 찾음
                    if area > max_area:
                        max_area = area
                        largest_vehicle_box = [x1, y1, width, height]
                        largest_vehicle_class_id = class_id

        # 가장 큰 박스가 있는 경우에만 그리기
        if largest_vehicle_box is not None:
            sx, sy, bw, bh = largest_vehicle_box
            color = (100, 100, 255)

            # 차량 바운딩 박스 그리기
            cv2.rectangle(frame, (sx, sy), (sx + bw, sy + bh), color, 2)

            # 거리 계산
            distance = (real_car_width * focal_length) / bw
            distance_text = f"Distance: {distance:.2f} meters"

            # 거리를 프레임에 표시
            cv2.putText(frame, distance_text, (sx, sy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            return frame, distance, largest_vehicle_box

        return frame, None, None


# if __name__ == "__main__":
#     # 비디오 파일 경로
#     video_path = 'sample1.avi'
#
#     # 비디오 캡처 객체 생성
#     cap = cv2.VideoCapture(video_path)
#     if not cap.isOpened():
#         print(f"Error: Failed to open video from {video_path}")
#         sys.exit(1)
#
#     # YOLOv8CarDetector 객체 생성
#     yolo = YOLOv8CarDetector('yolov8n.pt')
#
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             print("End of video or failed to read the frame.")
#             break
#
#         # 자동차 탐지 및 바운딩 박스 그리기
#         result_frame = yolo.detect_and_draw(frame)
#
#         # 결과 프레임 출력
#         cv2.imshow('Detected Cars', result_frame)
#         if cv2.waitKey(1) & 0xFF == ord('q'):  # 'q'를 눌러서 종료
#             break
#
#     cap.release()
#     cv2.destroyAllWindows()
