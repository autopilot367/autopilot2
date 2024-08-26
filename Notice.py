import cv2
import numpy as np

class Notice:
    def __init__(self):
        """ Init Thresholding """
        pass
    
    def rotate_image(self, image, angle):
        
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)

        M = cv2.getRotationMatrix2D(center, angle, 1.0)

        cos_theta = np.abs(M[0, 0])
        sin_theta = np.abs(M[0, 1])
        new_w = int((h * sin_theta) + (w * cos_theta))
        new_h = int((h * cos_theta) + (w * sin_theta))

        M[0, 2] += (new_w / 2) - center[0]
        M[1, 2] += (new_h / 2) - center[1]

        rotated_image = cv2.warpAffine(image, M, (new_w, new_h))
        return rotated_image

    def overlay_image(self, frame, overlay, position):

        (h, w) = overlay.shape[:2]
        (x, y) = position

        if y + h > frame.shape[0] or x + w > frame.shape[1]:
            return frame
        
        roi = frame[y:y+h, x:x+w]

        overlay_mask = overlay[:, :, 3] / 255.0
        roi = roi * (1 - overlay_mask[:, :, np.newaxis]) + overlay[:, :, :3] * overlay_mask[:, :, np.newaxis]

        frame[y:y+h, x:x+w] = roi
        return frame

    def combine(self, frame, angle):
        notice = Notice()

        overlay = cv2.imread("wheel_icon.png", cv2.IMREAD_UNCHANGED)  # 회전할 이미지 (투명도 채널이 포함될 수 있음)
        position = (50, 150)

        rotated_overlay = notice.rotate_image(overlay, angle)
        combined_frame = notice.overlay_image(frame, rotated_overlay, position)

        return combined_frame