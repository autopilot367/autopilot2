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
        rotated_image = cv2.warpAffine(image, M, (w, h))
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

        overlay = cv2.imread("wheel_icon_p.png", cv2.IMREAD_UNCHANGED)
        position = (500, 30)

        rotated_overlay = notice.rotate_image(overlay, angle)
        print(angle)
        combined_frame = notice.overlay_image(frame, rotated_overlay, position)

        return combined_frame
    
    def red_sign(self, img):
        warning = cv2.imread("warning_red.png", cv2.IMREAD_UNCHANGED)
        warning = cv2.resize(warning, (640, 360), interpolation=cv2.INTER_LINEAR)

        result = self.overlay_image(img, warning, (0,0))

        cv2.putText(result, "Warning!", (200, 300), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 0, 255), 1)

        return result