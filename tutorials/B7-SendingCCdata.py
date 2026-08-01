import numpy as np
import matplotlib.pyplot as plt
import rtmidi
import time
from rtmidi.midiconstants import CONTROL_CHANGE

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

def modulation_shape(repeat=1):
    """Function for modulation shape"""
    t = np.arange(0, 80, 0.1)
    amplitude = np.sin(t)

    plt.title("Modulation Shape")
    plt.xlabel('Time')
    plt.ylabel('Amplitude = sin(time)')
    plt.grid(True, which='both')
    plt.axhline(y=0, color='k')
    plt.plot(t[1:60], amplitude[1:60])
    plt.show()

    send_mod(amplitude, repeat)
    return amplitude

modulation_shape(1)
