import threading
import numpy as np
import matplotlib.pyplot as plt
import rtmidi
import time
from rtmidi.midiconstants import CONTROL_CHANGE
from scipy import signal

midiout = rtmidi.MidiOut()
midiout.open_port(1)

CHANNEL = 2
CC_NUM = 1
SPEED = 0.05
BPM = 60

def convert_range(value, in_min, in_max, out_min, out_max):
    """Converts a value from one range to another"""
    l_span = in_max - in_min
    r_span = out_max - out_min
    scaled_value = (value - in_min) / l_span
    scaled_value = out_min + (scaled_value * r_span)
    return np.round(scaled_value)

def play_melody(melody, bpm):
    """a function which loops over melody list"""
    for pitch, duration in melody:
        note_on = [0x90, pitch, 112]
        note_off = [0x80, pitch, 0]
        midiout.send_message(note_on)
        dur = duration_to_time_delay(duration, bpm)
        print(f'Sleep for{dur} seconds')
        time.sleep(dur)
        midiout.send_message(note_off)

def play_modulation(y, max_duration):
    """Send the modulation shape"""
    pause_duration = max_duration / y.size
    for v in y:
        v = convert_range(v, -1.0, 1.0, 0, 127)
        print(f"Mod:{v}")
        mod = ([CONTROL_CHANGE| CHANNEL, CC_NUM, v])
        midiout.send_message(mod)
        time.sleep(pause_duration)
def modulation_shape(shape: str, period: float, max_duration: float):
    x = np.arange(0, max_duration, 0.01)
    if shape == 'sine':
        y = np.sin(2 * np.pi/ period * x)
    elif shape == 'saw':
        y = signal.sawtooth(2 * np.pi / period * x)
    elif shape == 'square':
        y = signal.square(2 * np.pi / period * x)
    else:
        raise ValueError(f"Wave not supported {shape}")

    plt.plot(x, y)
    plt.ylabel(f"Amplitude = {shape} (time)")
    plt.xlabel('time')
    plt.axhline(y=0, color='Blue')
    plt.show()
    return y
def duration_to_time_delay(duration, bpm):
    if duration == 'w':
        factor = 4
    elif duration == 'h':
        factor = 2
    elif duration == 'q':
        factor = 1
    elif duration == 'e':
        factor = 0.5
    elif duration == 's':
        factor = 0.25
    else:
        assert False
    bps = bpm / 60
    return factor * bps

# List comprehension
def duration_of_melody(melody, bpm):
    return sum(duration_to_time_delay(duration, bpm) for _, duration in melody)

def main():
    melody = [(60, "e"), (62, "e"), (67, "q"), (62, "q"), (67, "q")] * 8
    dur = duration_of_melody(melody, BPM)
    print("Total duration of melody is", dur)
    modulation = modulation_shape("saw", period=1, max_duration=dur)
    t1= threading.Thread(target=play_melody, args=(melody,BPM))
    t2 = threading.Thread(target= play_modulation, args=(modulation, dur))
    t1.start()
    t2.start()
main ()
