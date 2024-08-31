import cv2

# 비디오 파일 경로
video_path = 'test_7.mp4'

# 비디오 캡처 객체 생성
cap = cv2.VideoCapture(video_path)

# 프레임 인덱스 설정
frame_index = 300
current_frame = 0

# 비디오가 열렸는지 확인
if not cap.isOpened():
    print("Error: Could not open video.")
else:
    while cap.isOpened():
        ret, frame = cap.read()

        # 비디오의 마지막이라면 루프 종료
        if not ret:
            print("Reached the end of the video or there was an error.")
            break

        # 5번째 프레임인지 확인
        if current_frame == frame_index:
            # 프레임을 640x360 크기로 리사이즈
            resized_frame = cv2.resize(frame, (640, 360))

            # 리사이즈된 프레임을 보여줌
            cv2.imshow('n_th Frame', resized_frame)
            cv2.waitKey(0)  # 키 입력을 기다림
            break

        # 현재 프레임 인덱스 증가
        current_frame += 1

# 비디오 캡처 객체 해제
cap.release()
cv2.destroyAllWindows()
