import sys
import bluetooth

devices = discover_devices()
devices

bd_addr = '[MAC-address for HC-06]'
port = 1
sock = bluetooth.BluetoothSocket(bluetooth.RFCOMM)
sock.connect((bd_addr,port))

sock.send("2".encode())