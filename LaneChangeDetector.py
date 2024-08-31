import cv2
import numpy as np
from LaneLines import LaneLines
from yolo_test import YOLO, find_only_front, get_largest_box
from ADAS_main import FindLaneLines

def getperspective_point(left_line, right_line, M_inv, y) :
        #print(M_inv)
        left_perspect_coor = np.float32([[left_line[0], y[0]]]).reshape(-1, 1, 2)
        left_original_coor = cv2.perspectiveTransform(left_perspect_coor, M_inv)
        right_perspect_coor = np.float32([[right_line[0], y[0]]]).reshape(-1, 1, 2)
        right_original_coor = cv2.perspectiveTransform(right_perspect_coor, M_inv)
        #print(left_original_coor, right_original_coor)
        return left_original_coor, right_original_coor
    
class LaneChangeDetector:
    def __init__(self):
        self.lane_detector = LaneLines()
        self.yolo = YOLO('yolov3.weights', 'yolov3.cfg', 'coco.names')

    def detect_lane_change(self, frame):
        # 1. 차선 검출 '핵심 코드' lanelines.fit_poly에서 ploty를 반환해야함. Findlanelines.forward에서 M_inv ploty(=y)를 반환해야 코드 작동
        findlinepoint = FindLaneLines()
        lane_img, left_line, right_line, M_inv, y = findlinepoint.forward(frame)
        left_line, right_line = getperspective_point(left_line, right_line, M_inv, y)
        left_line = int(left_line[0][0][0])
        right_line = int(right_line[0][0][0])
        
        # 2. YOLO를 사용하여 차량 검출
        roi_img = self.yolo.normalize_ROI(frame)
        boxes, confidences, class_ids = self.yolo.object_YOLO(roi_img)
        
        # 전방 차량만 찾기
        front_boxes, new_confidences = find_only_front(boxes, frame, confidences)
        
        # 가장 큰 차량 바운딩 박스 찾기
        if front_boxes:
            largest_box, largest_confidence = get_largest_box(front_boxes, new_confidences)

        lane_change_detected = []
        for box in largest_box:
            x, y, w, h = box
            centroid_x = x + (w / 2)
            
            print(centroid_x, left_line, right_line)
            
            # 좌우 차선과 차량의 위치 비교
            if left_line is not None and right_line is not None:

                if centroid_x - 0.45 * w < left_line:  # 차량이 왼쪽으로 이동
                    lane_change_detected = 0
                elif centroid_x + 0.45 * w > right_line:  # 차량이 오른쪽으로 이동
                    lane_change_detected = 1

        return lane_change_detected, largest_box, lane_img

if __name__ == "__main__":
    cap = cv2.VideoCapture('sample2.mp4')
    detector = LaneChangeDetector()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 차선 변경 여부 확인 및 바운딩 박스 반환
        lane_change, largest_box, lane_img = detector.detect_lane_change(frame)

        # 차선 변경 여부 텍스트 출력
        if lane_change == 0 :
            cv2.putText(lane_img, "Changing to left Lane", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                        1, (0, 0, 255), 2, cv2.LINE_AA)
        elif lane_change == 1 :
            cv2.putText(lane_img, "Changing to right Lane", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                        1, (0, 255, 0), 2, cv2.LINE_AA)

        # 결과 이미지 출력
        cv2.imshow('Frame', lane_img)
        if cv2.waitKey(10) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
