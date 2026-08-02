import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2
import math

mp_hands = mp.solutions.hands  # used for HAND_CONNECTIONS
mp_draw = mp.solutions.drawing_utils

# both hands track independently
latest_results = {"Left": None, "Right": None}

def calculate_pinch_distance(landmarks):
    thumb_tip = landmarks[4]
    index_tip = landmarks[8]
    distance = math.sqrt(
        (thumb_tip.x - index_tip.x) ** 2 +
        (thumb_tip.y - index_tip.y) ** 2
    )
    return distance

def result_callback(result, output_image, timestamp_ms):
    """
    Called automatically by MediaPipe after a frame is processed
    """
    latest_results["Left"] = None
    latest_results["Right"] = None

    if result.gestures and result.handedness:
        for gestures, handedness, landmarks in zip(result.gestures,
                                                     result.handedness,
                                                     result.hand_landmarks):
            raw_label = handedness[0].category_name
            label = "Left" if raw_label == "Right" else "Right"  #  as image is flipped
            gesture_name = gestures[0].category_name  # something like closed fist
            latest_results[label] = {
                "gesture": gesture_name,
                "landmarks": landmarks
            }


# gesture recognizer
base_options = mp_python.BaseOptions(model_asset_path="gesture_recognizer.task")
options = vision.GestureRecognizerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.LIVE_STREAM,
    num_hands=2,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5,
    result_callback=result_callback
)
recognizer = vision.GestureRecognizer.create_from_options(options)

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Webcam Error")
    exit()

while True:
    success, frame = cap.read()
    if not success:
        print("Ignoring empty frame from webcam")
        continue

    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

    timestamp_ms = int(cv2.getTickCount() / cv2.getTickFrequency() * 1000)
    recognizer.recognize_async(mp_image, timestamp_ms)

    # Draw whatever the most recent results were
    for label, data in latest_results.items():
        if data is None:
            continue

        landmarks = data["landmarks"]
        gesture_name = data["gesture"]

        # Rebuild landmarks
        proto_landmarks = landmark_pb2.NormalizedLandmarkList()
        proto_landmarks.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=lm.x, y=lm.y, z=lm.z)
            for lm in landmarks
        ])
        mp_draw.draw_landmarks(frame, proto_landmarks, mp_hands.HAND_CONNECTIONS)

        # gesture label
        wrist = landmarks[0]
        h, w, _ = frame.shape
        text_x, text_y = int(wrist.x * w), int(wrist.y * h) - 20
        cv2.putText(frame, f"{label}: {gesture_name}", (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        if label == "Right":
            pinch_distance = calculate_pinch_distance(landmarks)
            print(f"Pinch distance: {pinch_distance:.3f}")

            cv2.putText(frame, f"Pinch: {pinch_distance:.3f}", (text_x, text_y + 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    cv2.imshow("Gesture Recognition", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # for quitting
        break

cap.release()
cv2.destroyAllWindows()