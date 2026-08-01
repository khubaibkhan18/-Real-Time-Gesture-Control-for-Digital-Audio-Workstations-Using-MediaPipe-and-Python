import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils  # drawing module

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Webcam Error")
    exit()

while True:
    success, frame = cap.read()
    if not success:
        print("Ignoring empty frame from webcam")
        continue

    # mirroring frame cause of error
    frame = cv2.flip(frame, 1)

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    # loop through landmarks and handedness together
    if results.multi_hand_landmarks and results.multi_handedness:
        for hand_landmarks, handedness in zip(results.multi_hand_landmarks,
                                               results.multi_handedness):
            # Left and Right label
            label = handedness.classification[0].label

            wrist = hand_landmarks.landmark[0]
            print(f"{label} hand -> wrist x: {wrist.x:.3f}, y: {wrist.y:.3f}")

            # drawing on the hands
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    cv2.imshow("Hand Detection - Section 2", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()