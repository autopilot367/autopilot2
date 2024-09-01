import cv2
import TailDetector as tl
import numpy as np

# img1 img2를 비교하여 브레이크 상태 변화 분석
def CarBehaviour(img1,img2):
    # taildetector로 후미등 위치 탐지
    beh = ''
    bbox_img1, _ = tl.TailDetector(img1)
    bbox_img2, _ = tl.TailDetector(img2)
    print(f"bbox_img1: {bbox_img1}")
    print(f"bbox_img2: {bbox_img2}")

    if len(bbox_img1) == 0 or len(bbox_img2) == 0:
        print("bbox error... skipping")
        return None, None, None
    print("bbox_detected...")
    x, y, w, h =  bbox_img1[0][0], bbox_img1[0][1], bbox_img1[0][2],bbox_img1[0][3]
    right_light_img1 = img1[y:y + h, x:x + w]
    x, y, w, h = bbox_img1[1][0], bbox_img1[1][1], bbox_img1[1][2], bbox_img1[1][3]
    left_light_img1 = img1[y:y + h, x:x + w]


    x, y, w, h = bbox_img2[0][0], bbox_img2[0][1], bbox_img2[0][2], bbox_img2[0][3]
    right_light_img2 = img2[y:y + h, x:x + w]
    x, y, w, h = bbox_img2[1][0], bbox_img2[1][1], bbox_img2[1][2], bbox_img2[1][3]
    left_light_img2 = img2[y:y + h, x:x + w]

    status = False
    img1_yCrCb_r = cv2.cvtColor(right_light_img1, cv2.COLOR_BGR2YCrCb)
    img1_yCrCb_l = cv2.cvtColor(left_light_img1, cv2.COLOR_BGR2YCrCb)

    img2_yCrCb_r = cv2.cvtColor(right_light_img2, cv2.COLOR_BGR2YCrCb)
    img2_yCrCb_l = cv2.cvtColor(left_light_img2, cv2.COLOR_BGR2YCrCb)

    img1_Y_r,img1_Cr_r,img1_Cb_r = cv2.split(img1_yCrCb_r)
    img1_Y_l, img1_Cr_l, img1_Cb_l = cv2.split(img1_yCrCb_l)

    img2_Y_r,img2_Cr_r,img2_Cb_r = cv2.split(img2_yCrCb_r)
    img2_Y_l, img2_Cr_l, img2_Cb_l = cv2.split(img2_yCrCb_l)

    # 빨간색 성분(Cr) 이 200보다 큰 부분을 255, 아닌부분 0 반환
    ret, thresh1_r = cv2.threshold(img1_Cr_r, 140, 255, cv2.THRESH_BINARY) #test_7_2
    ret, thresh1_l = cv2.threshold(img1_Cr_l, 140, 255, cv2.THRESH_BINARY) #test_7_2

    ret2, thresh2_r = cv2.threshold(img2_Cr_r, 200, 255, cv2.THRESH_BINARY)
    ret, thresh2_l = cv2.threshold(img2_Cr_l, 200, 255, cv2.THRESH_BINARY)

    non1_r = cv2.countNonZero(thresh1_r)
    non1_l = cv2.countNonZero(thresh1_l)

    non2_r = cv2.countNonZero(thresh2_r)
    non2_l = cv2.countNonZero(thresh2_l)
    # 오른쪽 왼쪽 임계값이상인 픽셀의 개수의 합
    non1 = non1_r + non1_l
    non2 = non2_r + non2_l
    # 두 영상을 비교
    difference = [non1- non2, abs(non1 - non2)]
    # 후미등이 켜짐 : 픽셀 수 많음. 꺼짐 : 픽셀 수 적음
    # print(difference[1])
    # # 주행 중 정지하는 과정만 True반환
    # if difference[0] < 0 :
    #     status = True
    # else : status = False

    return difference[0], non1, non2

def count_pixels(frame):
    
    height, width = frame.shape[:2]
    total_pixels = height * width
    
    return total_pixels

if __name__ == "__main__":
    car_stop = cv2.imread('car_example_5.png', cv2.IMREAD_COLOR)
    car_not_stop = cv2.imread('car_example_5_1.png', cv2.IMREAD_COLOR)
    # x, y, w, h = 20, 150, 600, 250
    # car_stop = car_stop[y:y+h, x:x+w]
    # car_not_stop = car_not_stop[y:y+h, x:x+w]

    status = CarBehaviour(car_stop, car_not_stop)
    print(status)
    
    
    if cv2.waitKey(0) == 27: cv2.destroyAllWindows()
