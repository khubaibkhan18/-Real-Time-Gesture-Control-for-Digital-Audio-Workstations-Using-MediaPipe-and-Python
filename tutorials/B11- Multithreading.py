import time
import threading

def thread1():
    for i in range (10):
        print('1')
        time.sleep(0.01)
def thread2():
    for i in range(10):
        print('2')
        time.sleep(0.01)
my_thread1 = threading.Thread(target= thread1)
my_thread2 = threading.Thread(target= thread2)

my_thread2.start()
my_thread1.start()
