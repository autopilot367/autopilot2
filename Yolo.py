import cv2
import numpy as np
import sys


class YOLO:
    def __init__(self, yolo_weights, yolo_config, class_labels, conf_threshold=0.5, nms_threshold=0.4):

        self.yolo_weights = yolo_weights
        self.yolo_config = yolo_config
        self.class_labels = class_labels
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold

        self.net = cv2.dnn.readNet(self.yolo_weights, self.yolo_config)
        if self.net.empty():
            print('Net open failed!')
            sys.exit()

        with open(self.class_labels, 'rt') as f:
            self.classes = f.read().rstrip('\n').split('\n')

        self.colors = np.random.uniform(0, 255, size=(len(self.classes), 3))
        self.layer_names = self.net.getLayerNames()
        self.output_layers = [self.layer_names[i - 1] for i in self.net.getUnconnectedOutLayers()]

    def normalize_ROI(self, img, height_top_ratio=0.45):

        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        hist = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
        blur = cv2.GaussianBlur(hist, (5, 5), 1)
        height = blur.shape[0]
        width = blur.shape[1]
        height_top = int(height * height_top_ratio)
        polygons = np.array([
            [(40, height), (width - 40, height), (2 * width // 3, height_top), (width // 3, height_top)]
        ])
        mask = np.zeros_like(blur)
        cv2.fillPoly(mask, polygons, 255)
        masked_image = cv2.bitwise_and(blur, mask)
        roi_img = cv2.cvtColor(masked_image, cv2.COLOR_GRAY2RGB)
        return roi_img

    def object_YOLO(self, img):
        blob = cv2.dnn.blobFromImage(img, 1 / 255., (320, 320), swapRB=True)
        self.net.setInput(blob)
        outs = self.net.forward(self.output_layers)

        h, w = img.shape[:2]
        class_ids = []
        confidences = []
        boxes = []

        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > self.conf_threshold:
                    cx = int(detection[0] * w)
                    cy = int(detection[1] * h)
                    bw = int(detection[2] * w)
                    bh = int(detection[3] * h)
                    sx = int(cx - bw / 2)
                    sy = int(cy - bh / 2)
                    boxes.append([sx, sy, bw, bh])
                    confidences.append(float(confidence))
                    class_ids.append(int(class_id))

        return boxes, confidences, class_ids

    def draw_bounding_boxes(self, img, boxes, confidences, class_ids):
        indices = cv2.dnn.NMSBoxes(boxes, confidences, self.conf_threshold, self.nms_threshold)

        for i in indices:
            sx, sy, bw, bh = boxes[i]
            label = f'Vehicle : {confidences[i]:.2}'
            color = self.colors[class_ids[i]]
            cv2.rectangle(img, (sx, sy, bw, bh), color, 2)
            cv2.putText(img, label, (sx, sy - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)

        # t, _ = self.net.getPerfProfile()
        # label = 'Inference time: %.2f ms' % (t * 1000.0 / cv2.getTickFrequency())
        # cv2.putText(img, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
        #             0.7, (0, 0, 255), 1, cv2.LINE_AA)
        return img


# if __name__ == "__main__":
#
#     cap = cv2.VideoCapture('sample1.avi')
#     yolo = YOLO('yolov3.weights', 'yolov3.cfg', 'coco.names')
#
#     if not cap.isOpened():
#         print(f"Error: Failed to open video from {'sample1.avi'}")
#         exit(1)
#
#     while cap.isOpened():
#
#         ret, frame = cap.read()
#         if not ret:
#             print("Error: Failed to read frame from video. Exiting...")
#             break
#
#         # ROI 처리
#         roi_img = yolo.normalize_ROI(frame)
#
#         # YOLO 객체 검출
#         boxes, confidences, class_ids = yolo.object_YOLO(roi_img)
#
#         # 바운딩 박스 그리기
#         result_img = yolo.draw_bounding_boxes(frame, boxes, confidences, class_ids)
#
#         # 결과 이미지 출력
#         cv2.imshow('Result', result_img)
#         if cv2.waitKey(1) == 27:
#             break