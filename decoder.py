import socket

mysocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
mysocket.connect(('10.90.138.32', 10000))

i = 0
while i<100:
	i+=1
	msg = mysocket.recv(1024)
	print(str(msg, 'utf8'))
