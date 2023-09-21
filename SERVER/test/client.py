import random
import socket, select
from time import gmtime, strftime
from random import randint

image = "hello.png"

HOST = "192.168.56.1" 
PORT = 80

sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_address = (HOST, PORT)
sock.connect(server_address)

try:

    # open image
    myfile = open(image, 'rb')
    bytes = myfile.read()
    size = len(bytes)

    # send image size to server

    # send image to server
    sock.sendall(bytes)

    print ("Image successfully send to server")

    myfile.close()

finally:
    sock.close()