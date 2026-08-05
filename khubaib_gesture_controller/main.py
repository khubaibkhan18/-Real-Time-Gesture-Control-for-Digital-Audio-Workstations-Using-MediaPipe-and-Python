import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
from pythonosc.udp_client import SimpleUDPClient
import json
import rtmidi
import time
import math
import threading
from pythonosc.dispatcher import Dispatcher
from pythonosc.osc_server import BlockingOSCUDPServer

#FPS tracking, just for my own testing camera was laggy
fps_counter = 0
fps_timer = time.time()

# How many frames a Left-hand gesture needs to hold steady before it for gesture confidence
DEBOUNCE_FRAMES = 2

# Pointing_Up deletes a clip so need longer recognition time
DELETE_HOLD_SECONDS = 1.0

NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

# Tracks the debounce state for whatever gesture the Left hand is currently showing
candidate_gesture = {"Left": None}
candidate_count = {"Left": 0}
gesture_start_time = {"Left": None}
last_confirmed_gesture = {"Left": None}

# Holds the most recent gesture detected for both hands
latest_results = {"Left": None, "Right": None}
frame_timestamp = 0

# MIDIII
# Port 1 is my loopMIDI virtual port ("from_Python") so any notes are sent through to Ableton via this port

midiout = rtmidi.MidiOut()
print(midiout.get_ports())
midiout.open_port(1)

# talks to AbletonOSC, which listens on port 11000
ip = "127.0.0.1"
to_ableton = 11000
osc_client = SimpleUDPClient(ip, to_ableton)

# Ableton's "1. MIDI", "2. MIDI", "3. MIDI" (basically the tracks)
LOOP_TRACKS = [0, 1, 2]
current_track_index = 0

# which clip to record into
track_state = {
    track: {"next_clip_index": 0, "recorded_clips": []}
    for track in LOOP_TRACKS
}

# checks track state so victory can trigger opposite action
playing_state = {}


for t in LOOP_TRACKS:
    osc_client.send_message("/live/track/set/current_monitoring_state", [t, 1])
osc_client.send_message("/live/track/set/arm", [LOOP_TRACKS[0], 1])
print(f"Armed track {LOOP_TRACKS[0]} for recording")
print("NOTE: double check each track's Monitor is set to 'Auto' in Ableton if this doesn't take")

# Scale setup
MAJOR_PENTATONIC = [0, 2, 4, 7, 9]
MINOR_PENTATONIC = [0, 3, 5, 7, 10]

scale_info = {"root_note": None, "scale_name": None}
selected_clip_info = {"track": None, "clip": None}


def midi_to_note_name(midi_note):
    name = NOTE_NAMES[midi_note % 12]
    octave = midi_note // 12 - 1
    return f"{name}{octave}"


def on_root_note(address, *args):
    scale_info["root_note"] = args[0]


def on_scale_name(address, *args):
    scale_info["scale_name"] = args[0]


def on_selected_clip(address, *args):
    # so victory triggers selected clip
    selected_clip_info["track"] = args[0]
    selected_clip_info["clip"] = args[1]


""" Ableton sends its OSC replies back on port 11001, 
    so I need a small server running to actually receive them """

scale_dispatcher = Dispatcher()
scale_dispatcher.map("/live/song/get/root_note", on_root_note)
scale_dispatcher.map("/live/song/get/scale_name", on_scale_name)
scale_dispatcher.map("/live/view/get/selected_clip", on_selected_clip)

scale_server = BlockingOSCUDPServer((ip, 11001), scale_dispatcher)
scale_listener_thread = threading.Thread(target=scale_server.serve_forever, daemon=True)
scale_listener_thread.start()

# Ask Ableton what scale/root it's currently set to
osc_client.send_message("/live/song/get/root_note", [])
osc_client.send_message("/live/song/get/scale_name", [])
time.sleep(0.5)  # give it a moment to actually reply

if scale_info["root_note"] is not None:
    ROOT_NOTE = 60 + scale_info["root_note"]  # C4 as the base octave if Ableton doesn't respond properly
else:
    ROOT_NOTE = 60
    print("Could not read scale from Ableton - defaulting to C4")

# Sticking to pentatonic specifically as easy to mke mistakes and five notes correspond to five fingers

if scale_info["scale_name"] and "minor" in scale_info["scale_name"].lower():
    PENTATONIC_INTERVALS = MINOR_PENTATONIC
    print(f"Using MINOR pentatonic, root {ROOT_NOTE}")
else:
    PENTATONIC_INTERVALS = MAJOR_PENTATONIC
    print(f"Using MAJOR pentatonic, root {ROOT_NOTE}")

active_chord = {"Right": []}

with open("gesture_config.json") as f:
    gesture_config = json.load(f)

STANDARD_VELOCITY = 100

# Debounce for number of fingers detection
FINGER_DEBOUNCE_FRAMES = 3

# How much farther a fingertip needs to be from its knuckle
EXTENSION_RATIO = 1.25
THUMB_EXTENSION_RATIO = 1.15

EXPRESSION_CC = 11  # standard MIDI CC
last_expression_sent = {"Right": None}

# (tip landmark id, knuckle landmark id) for index, middle, ring, pinky.
# Thumb caused problems handling separately
FOUR_FINGER_LANDMARKS = [(8, 5), (12, 9), (16, 13), (20, 17)]

candidate_finger_count = {"Right": None}
candidate_finger_count_streak = {"Right": 0}
committed_finger_count = {"Right": 0}


def distance(a, b):
    return math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2)

# converting integers to MIDI understandable values
def convert_range(value, in_min, in_max, out_min, out_max):
    value = max(min(value, max(in_min, in_max)), min(in_min, in_max))
    l_span = in_max - in_min
    r_span = out_max - out_min
    scaled = (value - in_min) / l_span
    return out_min + (scaled * r_span)

# Only send message when change is meaningful instead of every frame, was causing lag
def send_expression(hand_label, value):
    value = int(max(0, min(127, value)))
    if last_expression_sent[hand_label] is None or abs(value - last_expression_sent[hand_label]) >= 2:
        midiout.send_message([0xB0, EXPRESSION_CC, value])
        last_expression_sent[hand_label] = value


def is_thumb_extended(landmarks):
    # The thumb folds SIDEWAYS across the palm,
    tip = landmarks[4]
    thumb_mcp = landmarks[2]
    ring_mcp = landmarks[17]

    tip_dist = distance(tip, ring_mcp)
    mcp_dist = distance(thumb_mcp, ring_mcp)
    return tip_dist > mcp_dist * THUMB_EXTENSION_RATIO


def count_extended_fingers(landmarks):
    wrist = landmarks[0]
    count = 0

    for tip_id, mcp_id in FOUR_FINGER_LANDMARKS:
        tip_dist = distance(landmarks[tip_id], wrist)
        mcp_dist = distance(landmarks[mcp_id], wrist)
        if tip_dist > mcp_dist * EXTENSION_RATIO:
            count += 1

    if is_thumb_extended(landmarks):
        count += 1

    return count


def build_chord(root_step):
    # Builds a triad using only notes from the pentatonic set itself
    degrees = len(PENTATONIC_INTERVALS)
    notes = []
    for offset in (0, 2, 4):
        idx = root_step + offset
        octave = idx // degrees
        degree = idx % degrees
        notes.append(ROOT_NOTE + PENTATONIC_INTERVALS[degree] + octave * 12)
    return notes


def update_chord(hand_label, notes, velocity):
    current = set(active_chord[hand_label])
    new = set(notes)

    if current == new:
        return

    notes_off = current - new
    notes_on = new - current

    for n in notes_off:
        midiout.send_message([0x80, n, 0])

    # Small gap between messages because Ableton was sending ghost notes (no sound)
    if notes_off and notes_on:
        time.sleep(0.005)

    for n in notes_on:
        midiout.send_message([0x90, n, velocity])

    active_chord[hand_label] = notes


def release_chord(hand_label):
    for n in active_chord[hand_label]:
        midiout.send_message([0x80, n, 0])
    active_chord[hand_label] = []


def draw_finger_status(frame, finger_count):
    if finger_count == 0:
        text = "Fingers: 0  (silent)"
    else:
        chord = build_chord(finger_count - 1)
        names = [midi_to_note_name(n) for n in chord]
        text = f"Fingers: {finger_count}  ->  {' - '.join(names)}"

    (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, 0.8, 1)
    pad = 14
    box_x1, box_y1 = 20, 20
    box_x2, box_y2 = box_x1 + text_w + pad * 2, box_y1 + text_h + pad * 2

    overlay = frame.copy()
    cv2.rectangle(overlay, (box_x1, box_y1), (box_x2, box_y2), (40, 40, 40), -1)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

    cv2.putText(frame, text, (box_x1 + pad, box_y2 - pad),
                cv2.FONT_HERSHEY_DUPLEX, 0.8, (210, 210, 210), 1, cv2.LINE_AA)


def draw_landmark_dots(frame, landmarks):
    # Just dots
    h, w, _ = frame.shape
    for lm in landmarks:
        x, y = int(lm.x * w), int(lm.y * h)
        cv2.circle(frame, (x, y), 4, (0, 220, 0), -1)


def get_current_track():
    return LOOP_TRACKS[current_track_index]


def start_recording():
    track = get_current_track()
    clip = track_state[track]["next_clip_index"]
    print(f"Thumb_Up -> START recording on track {track}, clip {clip}")
    osc_client.send_message("/live/clip_slot/fire", [track, clip])


def stop_recording():
    track = get_current_track()
    clip = track_state[track]["next_clip_index"]
    print(f"Thumb_Down -> STOP recording on track {track}, clip {clip} (loop begins)")
    osc_client.send_message("/live/clip_slot/fire", [track, clip])

    playing_state[(track, clip)] = True
    track_state[track]["recorded_clips"].append(clip)
    track_state[track]["next_clip_index"] += 1  # so the next Thumb_Up records a NEW clip


def toggle_last_clip():
    # Victory targets whatever clip is currently selected/highlighted
    osc_client.send_message("/live/view/get/selected_clip", [])
    time.sleep(0.05)

    track = selected_clip_info["track"]
    clip = selected_clip_info["clip"]

    if track is None or clip is None:
        print("Victory -> no clip currently selected in Ableton")
        return

    if playing_state.get((track, clip), False):
        print(f"Victory -> stopping selected clip {clip} on track {track}")
        osc_client.send_message("/live/clip/stop", [track, clip])
        playing_state[(track, clip)] = False
    else:
        print(f"Victory -> starting selected clip {clip} on track {track}")
        osc_client.send_message("/live/clip_slot/fire", [track, clip])
        playing_state[(track, clip)] = True


def delete_clip():
    # Only deletes the most recently recorded clip on the current track
    track = get_current_track()
    recorded = track_state[track]["recorded_clips"]

    if not recorded:
        print("Pointing_Up -> nothing to delete on this track")
        return

    clip = recorded.pop()
    print(f"Pointing_Up -> deleting clip {clip} on track {track}")
    osc_client.send_message("/live/clip_slot/delete_clip", [track, clip])

    playing_state.pop((track, clip), None)
    track_state[track]["next_clip_index"] = clip  # free up this slot for re-recording


def move_next_track():
    global current_track_index

    if current_track_index < len(LOOP_TRACKS) - 1:
        old_track = get_current_track()
        osc_client.send_message("/live/track/set/arm", [old_track, 0])

        current_track_index += 1
        new_track = get_current_track()
        osc_client.send_message("/live/track/set/arm", [new_track, 1])
        print(f"ILoveYou -> moved to track {new_track}, clip 0")
    else:
        print("ILoveYou -> already on the last track")


trigger_actions = {
    "Thumb_Up": start_recording,
    "Thumb_Down": stop_recording,
    "Victory": toggle_last_clip,
    "Pointing_Up": delete_clip,
    "ILoveYou": move_next_track,
}


def result_callback(result, output_image, timestamp_ms):
    # Called automatically by MediaPipe once it finishes processing a frame
    latest_results["Left"] = None
    latest_results["Right"] = None

    if result.gestures and result.handedness:
        for gestures, handedness, landmarks in zip(result.gestures,
                                                     result.handedness,
                                                     result.hand_landmarks):
            raw_label = handedness[0].category_name
            # We flip the camera frame for a natural mirror view
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

# DirectShow backend and small buffer for less lag
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

    small_frame = cv2.resize(frame, (480, 270))
    frame_rgb = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

    frame_timestamp += 1
    recognizer.recognize_async(mp_image, frame_timestamp)

    draw_finger_status(frame, committed_finger_count["Right"])

    for label, data in latest_results.items():
        if data is None:
            continue

        landmarks = data["landmarks"]
        gesture_name = data["gesture"]

        draw_landmark_dots(frame, landmarks)

        wrist_lm = landmarks[0]
        h, w, _ = frame.shape
        text_x, text_y = int(wrist_lm.x * w), int(wrist_lm.y * h) - 20
        cv2.putText(frame, f"{label}: {gesture_name}", (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # RIGHT HAND ONLY: notes, chords, and volume all live here.
        if label == "Right":
            raw_count = count_extended_fingers(landmarks)

            if raw_count == candidate_finger_count["Right"]:
                candidate_finger_count_streak["Right"] += 1
            else:
                candidate_finger_count["Right"] = raw_count
                candidate_finger_count_streak["Right"] = 1

            if (candidate_finger_count_streak["Right"] >= FINGER_DEBOUNCE_FRAMES
                    and raw_count != committed_finger_count["Right"]):
                committed_finger_count["Right"] = raw_count

                if raw_count == 0:
                    release_chord("Right")
                else:
                    chord = build_chord(raw_count - 1)
                    update_chord("Right", chord, STANDARD_VELOCITY)

            volume = convert_range(wrist_lm.y, 1.0, 0.0, 20, 127)
            send_expression("Right", volume)

        # LEFT HAND ONLY: discrete trigger gestures for loop recording, stopping,
        # toggle, moving to tracks and clips deleting
        if label == "Left":
            raw = gesture_name

            if raw == candidate_gesture["Left"]:
                candidate_count["Left"] += 1
            else:
                candidate_gesture["Left"] = raw
                candidate_count["Left"] = 1
                gesture_start_time["Left"] = time.time()
                last_confirmed_gesture["Left"] = None

            stable_enough = candidate_count["Left"] >= DEBOUNCE_FRAMES
            required_hold = DELETE_HOLD_SECONDS if raw == "Pointing_Up" else 0
            held_long_enough = (time.time() - gesture_start_time["Left"]) >= required_hold

            if stable_enough and held_long_enough and last_confirmed_gesture["Left"] != raw:
                if raw in trigger_actions:
                    trigger_actions[raw]()
                last_confirmed_gesture["Left"] = raw

    if latest_results["Right"] is None:
        release_chord("Right")
        committed_finger_count["Right"] = 0

    display_frame = cv2.resize(frame, (960, 540))
    cv2.imshow("Gesture Recognition", display_frame)

    if cv2.waitKey(1) & 0xFF == 27:  # Esc to quit
        break

cap.release()
cv2.destroyAllWindows()
release_chord("Right")  # don't leave a note hanging when the script closes