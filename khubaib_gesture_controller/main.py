import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2
from pythonosc.udp_client import SimpleUDPClient
import math
import json
import rtmidi
import time

mp_hands = mp.solutions.hands  # used for HAND_CONNECTIONS
mp_draw = mp.solutions.drawing_utils

fps_counter = 0
fps_timer = time.time()

# both hands track independently
latest_results = {"Left": None, "Right": None}
last_pinch_value = {"Right": None}
FREEZE_THRESHOLD = 0.75

previous_gesture = {"Left": None, "Right": None}
loop_is_playing = {}

frame_timestamp = 0

# --- MIDI setup ---
midiout = rtmidi.MidiOut()
print(midiout.get_ports())
midiout.open_port(1)

ip = "127.0.0.1"
to_ableton = 11000
osc_client = SimpleUDPClient(ip, to_ableton)

# Allows multiple recordings

LOOP_TRACKS = [0, 1, 2]
LOOP_CLIP = 0
current_track_index = 0
last_completed_track = None
# Arm the tracks from beginning
osc_client.send_message("/live/track/set/arm", [LOOP_TRACKS[0], 1])
print(f"Armed track {LOOP_TRACKS[0]} for recording (layer 1)")

# --- Scale setup ---
ROOT_NOTE = 60  # C4
PENTATONIC_INTERVALS = [0, 2, 4, 7, 9]
NUM_STEPS = len(PENTATONIC_INTERVALS) * 2

active_note = {"Right": None}

with open("gesture_config.json") as f:
    gesture_config = json.load(f)


def calculate_pinch_distance(landmarks, normalize=True):
    thumb_tip = landmarks[4]
    index_tip = landmarks[8]
    raw_distance = math.sqrt(
        (thumb_tip.x - index_tip.x) ** 2 +
        (thumb_tip.y - index_tip.y) ** 2 +
        (thumb_tip.z - index_tip.z) ** 2
    )

    if not normalize:
        return raw_distance

    wrist = landmarks[0]
    middle_knuckle = landmarks[9]
    hand_scale = math.sqrt(
        (wrist.x - middle_knuckle.x) ** 2 +
        (wrist.y - middle_knuckle.y) ** 2 +
        (wrist.z - middle_knuckle.z) ** 2
    )

    if hand_scale == 0:
        return raw_distance

    return raw_distance / hand_scale


def convert_range(value, in_min, in_max, out_min, out_max):
    value = max(min(value, max(in_min, in_max)), min(in_min, in_max))
    l_span = in_max - in_min
    r_span = out_max - out_min
    scaled = (value - in_min) / l_span
    return out_min + (scaled * r_span)


def fingertip_to_note(y):
    step = convert_range(y, 1.0, 0.0, 0, NUM_STEPS - 1)
    step = int(round(step))
    octave = step // len(PENTATONIC_INTERVALS)
    degree = step % len(PENTATONIC_INTERVALS)
    return ROOT_NOTE + PENTATONIC_INTERVALS[degree] + (octave * 12)


def pinch_to_velocity(pinch_distance):
    velocity = convert_range(pinch_distance, 0.12, 0.75, 0, 127)
    return int(max(0, min(127, velocity)))


def update_note(hand_label, note, velocity):
    current = active_note[hand_label]
    if current == note:
        return

    if current is not None:
        midiout.send_message([0x80, current, 0])

    midiout.send_message([0x90, note, velocity])
    active_note[hand_label] = note


def release_note(hand_label):
    current = active_note[hand_label]
    if current is not None:
        midiout.send_message([0x80, current, 0])
        active_note[hand_label] = None

# will start recording on selected track not a fixed one
def start_recording():
    track = get_current_track()
    print(f"Thumb_Up -> START recording on track {track}")
    osc_client.send_message("/live/clip_slot/fire", [track, LOOP_CLIP])

# will automatically stop recording of the chosen track and move forward if clip is available
def stop_recording():
    global current_track_index, last_completed_track

    track = get_current_track()
    print(f"Thumb_Down -> STOP recording on track {track} (loop begins)")
    osc_client.send_message("/live/clip_slot/fire", [track, LOOP_CLIP])
    loop_is_playing[track] = True
    last_completed_track = track
# to stop the right hand notes from going onto the next track
    osc_client.send_message("/live/track/set/arm", [track, 0])

    if current_track_index < len(LOOP_TRACKS) - 1:
        current_track_index += 1
        next_track = get_current_track()
        osc_client.send_message("/live/track/set/arm", [next_track, 1])
        print(f"Advanced to track {next_track} for next layer")
    else:
        print("All tracks used - no more layers available")

# toggle loop plays back recent loop now
def toggle_loop_playback():
    if last_completed_track is None:
        print("Victory -> no loop recorded yet")
        return

    track = last_completed_track
    if loop_is_playing.get(track, False):
        print(f"Victory -> stopping loop on track {track}")
        osc_client.send_message("/live/clip/stop", [track, LOOP_CLIP])
        loop_is_playing[track] = False
    else:
        print(f"Victory -> starting loop on track {track}")
        osc_client.send_message("/live/clip_slot/fire", [track, LOOP_CLIP])
        loop_is_playing[track] = True

# helper class for getting the track currently being played
def get_current_track():
    return LOOP_TRACKS[current_track_index]

trigger_actions = {
    "Thumb_Up": start_recording,
    "Thumb_Down": stop_recording,
    "Victory": toggle_loop_playback
}


def result_callback(result, output_image, timestamp_ms):
    latest_results["Left"] = None
    latest_results["Right"] = None

    if result.gestures and result.handedness:
        for gestures, handedness, landmarks in zip(result.gestures,
                                                     result.handedness,
                                                     result.hand_landmarks):
            raw_label = handedness[0].category_name
            label = "Left" if raw_label == "Right" else "Right"
            gesture_name = gestures[0].category_name
            latest_results[label] = {
                "gesture": gesture_name,
                "landmarks": landmarks
            }


base_options = mp_python.BaseOptions(model_asset_path="gesture_recognizer.task")
options = vision.GestureRecognizerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.LIVE_STREAM,
    num_hands=2,
    min_hand_detection_confidence=0.3,
    min_hand_presence_confidence=0.3,
    min_tracking_confidence=0.3,
    result_callback=result_callback
)
recognizer = vision.GestureRecognizer.create_from_options(options)

# NEW: DirectShow backend + minimal buffer, both reduce camera-side lag on Windows
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

if not cap.isOpened():
    print("Webcam Error")
    exit()

while True:
    success, frame = cap.read()
    if not success:
        print("Ignoring empty frame from webcam")
        continue

    fps_counter += 1
    if time.time() - fps_timer >= 1.0:
        print(f"FPS: {fps_counter}")
        fps_counter = 0
        fps_timer = time.time()

    frame = cv2.flip(frame, 1)

    # NEW: downscale specifically for MediaPipe processing - big speed win.
    # Landmark coordinates are normalized (0.0-1.0) so they still map
    # correctly onto the full-size 'frame' below for drawing/display.
    small_frame = cv2.resize(frame, (480, 270))
    frame_rgb = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

    frame_timestamp += 1
    recognizer.recognize_async(mp_image, frame_timestamp)

    for label, data in latest_results.items():
        if data is None:
            continue

        landmarks = data["landmarks"]
        gesture_name = data["gesture"]

        proto_landmarks = landmark_pb2.NormalizedLandmarkList()
        proto_landmarks.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=lm.x, y=lm.y, z=lm.z)
            for lm in landmarks
        ])
        mp_draw.draw_landmarks(frame, proto_landmarks, mp_hands.HAND_CONNECTIONS)

        wrist = landmarks[0]
        h, w, _ = frame.shape
        text_x, text_y = int(wrist.x * w), int(wrist.y * h) - 20
        cv2.putText(frame, f"{label}: {gesture_name}", (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        if label == "Right":
            middle_tip = landmarks[12]
            note = fingertip_to_note(middle_tip.y)

            raw_pinch = calculate_pinch_distance(landmarks)
            if raw_pinch < FREEZE_THRESHOLD:
                last_pinch_value["Right"] = raw_pinch
                pinch_distance = raw_pinch
            else:
                pinch_distance = last_pinch_value["Right"]

            if pinch_distance is not None:
                velocity = pinch_to_velocity(pinch_distance)
                update_note("Right", note, velocity)

                cv2.putText(frame, f"Pinch: {pinch_distance:.3f}", (text_x, text_y + 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        if label == "Left":
            if gesture_name != previous_gesture["Left"]:
                if gesture_name in trigger_actions:
                    trigger_actions[gesture_name]()

            previous_gesture["Left"] = gesture_name

    if latest_results["Right"] is None:
        release_note("Right")

    display_frame = cv2.resize(frame, (640, 360))
    cv2.imshow("Gesture Recognition", display_frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
release_note("Right")