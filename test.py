import socket
 
serverMACAddress = '0d:e8:69:46:01:98'
port = 4
s = socket.socket(socket.AF_BLUETOOTH, socket.SOCK_STREAM, socket.BTPROTO_RFCOMM)
s.connect((serverMACAddress,port))
s.send(bytes("A"))
s.close()