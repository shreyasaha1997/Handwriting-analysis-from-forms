import threading
import time
def f(a):
    print('hello')
    time.sleep(5)
    print("ddd")
    time.sleep(5)

download_thread = threading.Thread(target=f,kwargs={'a': 1})
download_thread.start()
print('hola')
