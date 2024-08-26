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

    def detect_and_draw(self, frame):
        # YOLOv8 모델을 사용하여 프레임에서 객체 탐지
        results = self.model(frame)

        # 결과 프레임에 바운딩 박스 및 라벨을 그리기
        annotated_frame = results[0].plot()

        return annotated_frame

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
