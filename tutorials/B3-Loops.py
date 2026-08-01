import rtmidi
import time

midiout = rtmidi.RtMidiOut()
ports = midiout.getPortCount()
print(ports)

midiout.openPort(1)
tempo = 0.3

try:
    for bar in range (4):
        for note in range(4):
            note_on = rtmidi.MidiMessage.noteOn(0x90, 60, 100)
            note_off = rtmidi.MidiMessage.noteOff(0x80, 60)
            midiout.sendMessage(note_on)
            time.sleep(tempo)
            midiout.sendMessage(note_off)

        for note in range (6):
            note_on = rtmidi.MidiMessage.noteOn(0x90, 62, 100)
            note_off = rtmidi.MidiMessage.noteOff(0x80, 62)

            midiout.sendMessage(note_on)
            time.sleep(tempo)
            midiout.sendMessage(note_off)
        for note in range(4):
            note_on = rtmidi.MidiMessage.noteOn(0x90, 63, 100)
            note_off = rtmidi.MidiMessage.noteOff(0x80, 63)
            midiout.sendMessage(note_on)
            time.sleep(tempo)
            midiout.sendMessage(note_off)

        for note in range(6):
            note_on = rtmidi.MidiMessage.noteOn(0x90, 60, 100)
            note_off = rtmidi.MidiMessage.noteOff(0x80, 60)
            midiout.sendMessage(note_on)
            time.sleep(tempo)
            midiout.sendMessage(note_off)

finally:
    midiout.closePort()
    del midiout

