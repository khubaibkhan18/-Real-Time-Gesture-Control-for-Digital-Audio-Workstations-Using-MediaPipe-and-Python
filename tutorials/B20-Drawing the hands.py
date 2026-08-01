import cv2
import time
import mediapipe as mp
import numpy as np
import rtmidi
from rtmidi.midiconstants import CONTROL_CHANGE

midiout = rtmidi.MidiOut()
print(midiout.get_ports())
midiout.open_port(1)

cap = cv2.VideoCapture(0)
# 1080p gives MediaPipe a lot more pixels to process per frame, which adds
# latency. 1280x720 is a good middle ground between detection quality and
# responsiveness. Bump this back up to 1920x1080 if your machine can keep up.
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

actual_w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
actual_h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
print(f"Requested 1280x720, camera actually gave: {int(actual_w)}x{int(actual_h)}")

mpHands = mp.solutions.hands
hands = mpHands.Hands(static_image_mode=False,
                       max_num_hands=2,
                       min_detection_confidence=0.5,
                       min_tracking_confidence=0.5)
mpDraw = mp.solutions.drawing_utils


def convert_range(value, in_min, in_max, out_min, out_max):
    """Converts a value from one range to another"""
    l_span = in_max - in_min
    r_span = out_max - out_min
    scaled_value = (value - in_min) / l_span
    scaled_value = out_min + (scaled_value * r_span)
    return int(np.round(scaled_value))


# --- Note sustain state -----------------------------------------------
# Tracks the currently-held note per physical hand label ("Left"/"Right").
# A note stays on as long as the hand is present and the pitch hasn't
# changed. It's released the moment the hand disappears or the pitch moves.
active_notes = {"Left": None, "Right": None}

# --- Smoothing + debounce state for the right-hand (note) control ------
# Raw pinky-y is noisy frame-to-frame, which causes the mapped note to
# flicker between neighbouring semitones even when your hand is basically
# still. We fix this two ways:
#   1. Exponential smoothing on the y value itself (SMOOTHING closer to 1
#      = smoother but more lag; closer to 0 = snappier but more jitter).
#   2. A debounce: a new note only "wins" once it has been the candidate
#      for DEBOUNCE_FRAMES consecutive frames, so a single noisy frame
#      can't retrigger a note.
SMOOTHING = 0.7          # 0.0 = no smoothing, 0.95 = very smooth/laggy
DEBOUNCE_FRAMES = 4      # consecutive frames a new note must "win" before it plays
MIN_NOTE_HOLD = 0.08     # seconds - minimum time a note stays before it can retrigger

smoothed_y = {"Right": None}
candidate_pitch = {"Right": None}
candidate_count = {"Right": 0}
last_note_change_time = {"Right": 0.0}


def update_note(hand_label, pitch):
    pitch = int(np.clip(pitch, 0, 127))
    current = active_notes[hand_label]

    if current == pitch:
        candidate_count[hand_label] = 0
        return  # already sustaining this exact note, nothing to do

    # Require the new pitch to be the "winning" candidate for several
    # consecutive frames before we actually switch notes.
    if candidate_pitch[hand_label] == pitch:
        candidate_count[hand_label] += 1
    else:
        candidate_pitch[hand_label] = pitch
        candidate_count[hand_label] = 1

    enough_frames = candidate_count[hand_label] >= DEBOUNCE_FRAMES
    enough_time = (time.time() - last_note_change_time[hand_label]) >= MIN_NOTE_HOLD

    if not (enough_frames and enough_time):
        return

    if current is not None:
        midiout.send_message([0x80, current, 0])  # release old note

    midiout.send_message([0x90, pitch, 112])  # sound new note
    active_notes[hand_label] = pitch
    last_note_change_time[hand_label] = time.time()
    candidate_count[hand_label] = 0


def release_note(hand_label):
    current = active_notes[hand_label]
    if current is not None:
        midiout.send_message([0x80, current, 0])
        active_notes[hand_label] = None


def send_mod(cc=1, value=0):
    value = int(np.clip(value, 0, 127))
    mod1 = [CONTROL_CHANGE | 0, cc, value]
    midiout.send_message(mod1)


while True:
    success, img = cap.read()
    if not success:
        break

    img = cv2.flip(img, 1)  # mirror for a natural selfie-view
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    results = hands.process(imgRGB)

    h, w, c = img.shape

    seen_this_frame = set()

    if results.multi_hand_landmarks and results.multi_handedness:
        for hand_landmarks, handedness in zip(results.multi_hand_landmarks,
                                               results.multi_handedness):
            # Because we flip the image ourselves before processing (mirror
            # view), MediaPipe's Left/Right label matches your actual
            # physical hand as you see it in the mirror.
            label = handedness.classification[0].label  # "Left" or "Right"
            seen_this_frame.add(label)

            pink_x = hand_landmarks.landmark[mpHands.HandLandmark.PINKY_TIP].x
            pink_y = hand_landmarks.landmark[mpHands.HandLandmark.PINKY_TIP].y
            px = pink_x * w

            if label == "Left":
                v1 = convert_range(pink_y, 1.0, 0.0, 0, 127)
                send_mod(1, v1)
                print(f"[Left hand] MIDI CC -> {v1}  (x={px:.0f}, y={pink_y:.2f})")
            else:
                if smoothed_y["Right"] is None:
                    smoothed_y["Right"] = pink_y
                else:
                    smoothed_y["Right"] = (SMOOTHING * smoothed_y["Right"]
                                            + (1 - SMOOTHING) * pink_y)

                v2 = convert_range(smoothed_y["Right"], 1.0, -1.0, 60, 92)
                update_note(label, v2)
                print(f"[Right hand] MIDI Note -> {active_notes[label]}  (x={px:.0f}, y={pink_y:.2f})")

            mpDraw.draw_landmarks(img, hand_landmarks, mpHands.HAND_CONNECTIONS)

    # Release the sustained note for any hand that vanished from this frame
    for hand_label in ("Left", "Right"):
        if hand_label not in seen_this_frame:
            release_note(hand_label)
            if hand_label == "Right":
                smoothed_y["Right"] = None
                candidate_pitch["Right"] = None
                candidate_count["Right"] = 0

    cv2.putText(img, f"{w}x{h}", (10, 70), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 3)
    cv2.imshow("Khubaib", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Make sure nothing is left stuck on when we quit
for hand_label in ("Left", "Right"):
    release_note(hand_label)

cap.release()
cv2.destroyAllWindows()