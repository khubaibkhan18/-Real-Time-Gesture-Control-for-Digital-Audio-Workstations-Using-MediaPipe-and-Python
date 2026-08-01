import numpy as np
import matplotlib.pyplot as plt
import rtmidi
import time
from rtmidi.midiconstants import CONTROL_CHANGE
from scipy import signal
import sys
midiout = rtmidi.MidiOut()
midiout.open_port(1)

CHANNEL = 0
CC_NUM = 75
SPEED = 0.05

def convert_range(value, in_min, in_max, out_min, out_max):
    """Converts a value from one range to another"""
    l_span = in_max - in_min
    r_span = out_max - out_min
    scaled_value = (value - in_min) / l_span
    scaled_value = out_min + (scaled_value * r_span)
    return np.round(scaled_value)

def send_mod(amplitude, repeat):
    """Converts amplitude values and sends them as MIDI CC messages"""
    scaled = []
    for amp in amplitude:
        val = int(convert_range(amp, -1, 1, 0, 127))  # convert to int
        scaled.append(val)
    for _ in range(repeat):
        for value in scaled:
            mod = [CONTROL_CHANGE | CHANNEL, CC_NUM, value]
            midiout.send_message(mod)   # snake_case
            time.sleep(SPEED)

BPM = 90
def modulation_shape(shape: str, period: float, max_duration: float):
    x = np.arange(0, max_duration, 0.01)
    if shape == 'sine':
        y = np.sin(2 * np.pi/ period * x)
    elif shape == 'saw':
        y = signal.sawtooth(2 * np.pi / period * x)
    elif shape == 'square':
        y = signal.square(2 * np.pi / period * x)
    else:
        print('Wave not supported')
        sys.exit()
    plt.plot(x, y)
    plt.ylabel(f"Amplitude = {shape} (time)")
    plt.xlabel('time')
    plt.axhline(y=0, color='Blue')
    plt.show()
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



"""def duration_of_melody(melody, bpm):
    t = 0
    for _, duration in melody:
        t += duration_to_time_delay(duration, bpm)
        print(f"We need to wait {t} seconds")
    return t"""

# List comprehension
def duration_of_melody(melody, bpm):
    return sum(duration_to_time_delay(duration, bpm) for _, duration in melody)

def main():
    melody = [(60, "e"), (62, "e"), (67, "q"), (62, "q"), (67, "q")] * 8
    dur = duration_of_melody(melody, BPM)
    print("Total duration of melody is", dur)
main ()
