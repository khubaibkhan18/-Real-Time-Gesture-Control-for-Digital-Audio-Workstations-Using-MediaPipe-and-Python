import rtmidi
import time

midiout = rtmidi.RtMidiOut()

portCount = midiout.getPortCount()
print(portCount)
portName = [midiout.getPortName(i) for i in range(portCount)]
print(portName)

midiout.openPort(1)

note_on = rtmidi.MidiMessage.noteOn(0x90, 60, 100)
note_off = rtmidi.MidiMessage.noteOff(0x80, 60)

midiout.sendMessage(note_on)
time.sleep(0.2)
midiout.sendMessage(note_off)
time.sleep(0.3)
midiout.sendMessage(note_on)
time.sleep(0.4)
midiout.sendMessage(note_off)
time.sleep(0.5)
midiout.sendMessage(note_on)
time.sleep(0.6)
midiout.sendMessage(note_off)
time.sleep(0.7)
midiout.sendMessage(note_on)
time.sleep(0.8)
midiout.sendMessage(note_off)
time.sleep(0.9)
midiout.sendMessage(note_on)
time.sleep(1.0)
midiout.sendMessage(note_off)
time.sleep(1.0)
midiout.sendMessage(note_on)
time.sleep(0.3)
midiout.sendMessage(note_off)
time.sleep(0.5)
midiout.closePort()
