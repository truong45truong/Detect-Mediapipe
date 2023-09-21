import socket 
soc =socket.socket(socket.AF_INET, socket.SOCK_STREAM)
soc.connect(("192.168.220.212", 7801))
soc.sendall(b'cact')