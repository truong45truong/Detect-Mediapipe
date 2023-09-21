import socket
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from os import error
import threading

HOST = socket.gethostbyname(socket.gethostname())
PORT = 80

SERVER = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

try:
    SERVER.bind((HOST,PORT))
    print(f'* Running on http://{HOST}:{PORT}')
except socket.error as e:
    print(f'socket error: {e}')
    print('socket error: %s' %(e))


def _start():
    SERVER.listen()
    while True:
        conn, addr = SERVER.accept()
        print(conn,addr)
        thread = threading.Thread(target=_handle, args=(conn, addr))
        thread.start()

def _handle(conn,addr):
    while True:
        data = conn.recv(4096)
        if not data: break
        print(data.decode())
        conn.close()
        break


if __name__ == '__main__':
    _start()

