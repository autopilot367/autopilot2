import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


class Notice:
    def __init__(self):
        """ Init Thresholding """
        pass

    def rotate_image(self, image, angle):

        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)

        M = cv2.getRotationMatrix2D(center, angle * 1, 1.0)
        rotated_image = cv2.warpAffine(image, M, (w, h))
        return rotated_image

    def overlay_image(self, frame, overlay, position):

        (h, w) = overlay.shape[:2]
        (x, y) = position

        if y + h > frame.shape[0] or x + w > frame.shape[1]:
            return frame

        roi = frame[y:y + h, x:x + w]

        overlay_mask = overlay[:, :, 3] / 255.0
        roi = roi * (1 - overlay_mask[:, :, np.newaxis]) + overlay[:, :, :3] * overlay_mask[:, :, np.newaxis]

        frame[y:y + h, x:x + w] = roi
        return frame

    def combine(self, frame, angle):
        notice = Notice()

        overlay = cv2.imread("icon_wheel.png", cv2.IMREAD_UNCHANGED)
        position = (500, 30)

        rotated_overlay = notice.rotate_image(overlay, angle)
        print(angle)
        combined_frame = notice.overlay_image(frame, rotated_overlay, position)

        # 각도 표시
        combined_PIL = Image.fromarray(cv2.cvtColor(combined_frame, cv2.COLOR_BGR2RGB))

        draw = ImageDraw.Draw(combined_PIL)
        text = f"{int(angle)}°"
        font = ImageFont.truetype("arial.ttf", 20)

        bbox = draw.textbbox((0, 0), text, font=font)
        text_w, text_h = bbox[2] - bbox[0], bbox[3] - bbox[1]
        draw.text((550 - (text_w // 2), 130), text, font=font, fill=(255, 255, 255))

        combined_frame = cv2.cvtColor(np.array(combined_PIL), cv2.COLOR_RGB2BGR)

        return combined_frame

    def red_sign(self, img):
        warning = cv2.imread("warning_red.png", cv2.IMREAD_UNCHANGED)
        warning = cv2.resize(warning, (640, 360), interpolation=cv2.INTER_LINEAR)

        result = self.overlay_image(img, warning, (0, 0))

        cv2.putText(result, "Collision Warning", (190, 320), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 1,
                    lineType=cv2.LINE_AA)

        return result

    def green_sign(self, img, deviation):
        if deviation < 0 :
            warning1 = cv2.imread("warning_org_left.png", cv2.IMREAD_UNCHANGED)
            warning1 = cv2.resize(warning1, (640, 360), interpolation=cv2.INTER_LINEAR)
            result = self.overlay_image(img, warning1, (0, 0))
        else :
            warning2 = cv2.imread("warning_org_right.png", cv2.IMREAD_UNCHANGED)
            warning2 = cv2.resize(warning2, (640, 360), interpolation=cv2.INTER_LINEAR)
            result = self.overlay_image(img, warning2, (0, 0))
        # cv2.putText(img, f"deviation: {deviation}", (180, 330), cv2.FONT_HERSHEY_DUPLEX, 0.75, (255, 255, 255), 1,
        #             lineType=cv2.LINE_AA)
        # cv2.putText(result, "Lane Departure Warning", (160, 340), cv2.FONT_HERSHEY_DUPLEX, 0.75, (255, 255, 255), 1,
        #             lineType=cv2.LINE_AA)

        return result

    def blue_sign(self, img, lane_change):
        warning = cv2.imread("warning_blue.png", cv2.IMREAD_UNCHANGED)
        warning = cv2.resize(warning, (640, 360), interpolation=cv2.INTER_LINEAR)

        result = self.overlay_image(img, warning, (0, 0))

        if lane_change == 0:
            cv2.putText(result, "Changing to left Lane", (190, 320), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 1,
                        lineType=cv2.LINE_AA)
        elif lane_change == 1:
            cv2.putText(result, "Changing to right Lane", (190, 320), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 1,
                        lineType=cv2.LINE_AA)

        return result
# import cv2
# import numpy as np
#
# class Notice:
#     def __init__(self):
#         """ Init Thresholding """
#         pass
#
#     def rotate_image(self, image, angle):
#
#         (h, w) = image.shape[:2]
#         center = (w // 2, h // 2)
#
#         M = cv2.getRotationMatrix2D(center, angle, 1.0)
#         rotated_image = cv2.warpAffine(image, M, (w, h))
#         return rotated_image
#
#     def overlay_image(self, frame, overlay, position):
#
#         (h, w) = overlay.shape[:2]
#         (x, y) = position
#
#         if y + h > frame.shape[0] or x + w > frame.shape[1]:
#             return frame
#
#         roi = frame[y:y+h, x:x+w]
#
#         overlay_mask = overlay[:, :, 3] / 255.0
#         roi = roi * (1 - overlay_mask[:, :, np.newaxis]) + overlay[:, :, :3] * overlay_mask[:, :, np.newaxis]
#
#         frame[y:y+h, x:x+w] = roi
#         return frame
#
#     def combine(self, frame, angle):
#         notice = Notice()
#
#         overlay = cv2.imread("wheel_icon_p.png", cv2.IMREAD_UNCHANGED)
#         position = (500, 30)
#
#         rotated_overlay = notice.rotate_image(overlay, angle)
#         print(angle)
#         combined_frame = notice.overlay_image(frame, rotated_overlay, position)
#
#         return combined_frame
#
#     def red_sign(self, img):
#         warning = cv2.imread("warning_red.png", cv2.IMREAD_UNCHANGED)
#         warning = cv2.resize(warning, (640, 360), interpolation=cv2.INTER_LINEAR)
#
#         result = self.overlay_image(img, warning, (0,0))
#
#         cv2.putText(result, "Warning!", (200, 300), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 0, 255), 1)
#
#         return result