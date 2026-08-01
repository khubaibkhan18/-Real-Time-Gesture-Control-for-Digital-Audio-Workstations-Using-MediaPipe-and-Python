import cv2
import time
import mediapipe as mp

cap = cv2.VideoCapture(0)

mpHands = mp.solutions.hands
hands = mpHands.Hands(static_image_mode=False,
                       max_num_hands=2,
                       min_detection_confidence=0.5,
                       min_tracking_confidence=0.5)
mpDraw = mp.solutions.drawing_utils

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    results = hands.process(imgRGB)
    if results.multi_hand_landmarks:
        h, w, c = img.shape
        print(results.multi_hand_landmarks)
    fps = 1
    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 10)
    cv2.imshow("Khubaib", img)
    cv2.waitKey(fps)