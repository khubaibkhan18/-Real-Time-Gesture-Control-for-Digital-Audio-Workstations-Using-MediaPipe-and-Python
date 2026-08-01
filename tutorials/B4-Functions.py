import rtmidi
import time

midiout = rtmidi.RtMidiOut()
midiout.openPort(1)
def send_notes(pitch, repeat):
    for note in range(repeat):
        note_on = rtmidi.MidiMessage.noteOn(0x90, pitch, 80)
        note_off = rtmidi.MidiMessage.noteOff(0x80, pitch)
        midiout.sendMessage(note_on)
        time.sleep(0.2)
        midiout.sendMessage(note_off)

try:
    for i in range(10):
        send_notes(60, 2)
        send_notes(63,2)
        send_notes(60,2)
        send_notes(65,2)
        send_notes(60,2)
        send_notes(67,2)
finally:
    midiout.closePort()
    del midiout