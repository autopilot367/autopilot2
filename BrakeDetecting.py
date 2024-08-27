import cv2
from yolo_test import *
import CarBehaviour as cb

def roi_for_tail_detect(frame):
    
    yolo = YOLO('yolov3.weights', 'yolov3.cfg', 'coco.names')
    roi_img = yolo.normalize_ROI(frame)
    # YOLO 객체 검출
    boxes, confidences, class_ids = yolo.object_YOLO(roi_img)
    boxes, confidences = find_only_front(boxes, frame, confidences)
    boxes, confidences = get_largest_box(boxes, confidences)
    x, y, w, h = boxes[0]
    roi_frame = frame[y+int(0.3*h):y+int(0.8*h), x+int(0.08*w):x+int(0.92*w)]
    
    return roi_frame


# if __name__ == "__main__":
#     # YOLO 모델 로드
#     yolo = YOLO('yolov3.weights', 'yolov3.cfg', 'coco.names')
#     # 비디오 파일 경로
#     video_path = 'road_10.mp4'

#     # 비디오 파일 열기
#     cap = cv2.VideoCapture(video_path)

#     # FPS(초당 프레임 수) 얻기
#     fps = cap.get(cv2.CAP_PROP_FPS)

#     # 0.5초에 해당하는 프레임 간격 계산
#     frame_interval = int(fps * 0.2)

#     # 첫 번째 프레임 읽기
#     ret, previous_frame = cap.read()

#     # 첫 번째 프레임이 유효한 경우에만 처리
#     if ret:
#         previous_frame = roi_for_tail_detect(previous_frame)

#     while ret:
#         # 0.5초마다 프레임 읽기
#         cap.set(cv2.CAP_PROP_POS_FRAMES, cap.get(cv2.CAP_PROP_POS_FRAMES) + frame_interval)
#         ret, current_frame = cap.read()

#         if not ret:
#             break

#         current_frame = roi_for_tail_detect(current_frame)

#         # 첫 번째 프레임과 현재 프레임의 후미등 비교
#         difference = cb.CarBehaviour(previous_frame, current_frame)
#         print(difference / cb.count_pixels(previous_frame))
#         if difference / cb.count_pixels(previous_frame) >= 0.001 : print(True)
#         else : print(False)

#         # 현재 프레임을 다음 비교를 위한 이전 프레임으로 설정
#         previous_frame = current_frame

#         # ESC를 누르면 루프 중단
#         if cv2.waitKey(10) == 27:
#             break

#     # 모든 작업 완료 후 자원 해제
#     cv2.destroyAllWindows()
#     cap.release()
