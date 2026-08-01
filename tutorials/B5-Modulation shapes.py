import numpy as np
import matplotlib.pyplot as plt
"""this is a function """
def convert_range(value, in_min, in_max, out_min, out_max):
    l_span = in_max - in_min
    r_span = out_max-out_min
    scaled_value = (value - in_min)/ l_span
    scaled_value = out_min + (scaled_value * r_span)
    return np.round(scaled_value)

def modulation_shape(repeat=1):
    """Function for modulation shape"""
    t = np.arange(0, 80, 0.1)
    # start value, end value, interval (time)
    amplitude = np.sin(t)
    plt.title("Modulation Shape")
    plt.xlabel('Time')
    plt.ylabel('Amplitude= sin(time)')
    plt.grid(True, which='both')
    plt.axhline(y=0, color='k')
    plt.plot(t[1:60], amplitude[1:60])
    plt.show()
    return amplitude

amp = modulation_shape(1)


converted_amplitude = []
for number in amp:
    result = convert_range(number, -1.0, 1, 0, 127)
    converted_amplitude.append(result)
converted_amplitude = [int(x) for x in converted_amplitude]
print(converted_amplitude)

"""this_is_a_list= [1, 2, 3, 4, 5]
print(this_is_a_list[2:])
#since it starts from 0 this will print 3-5. List slicing."""
""" this is an appended list
fruits= []
print (fruits)
fruits.append('Banana')
print(fruits) """
