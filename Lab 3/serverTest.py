# Reminder: This is a comment. The first line imports a default library "socket" into Python.
# You don’t install this. The second line is initialization to add TCP/IP protocol to the endpoint.
import socket
serv = socket.socket(socket.AF_INET, socket.SOCK_STREAM) 

host_name = socket.gethostname()
print(host_name)
thingy = socket.gethostbyname(host_name)
print(thingy)

# Assigns a port for the server that listens to clients connecting to this port.
serv.bind(('169.254.112.183', 8080))
serv.listen(5)
while True:
    conn, addr = serv.accept()
    from_client = ''
    while True:
        data = conn.recv(4096)
        if not data: break
        from_client += data.decode()
        print(from_client)
        conn.send("I am SERVER\n".encode())
    conn.close()
    print('client disconnected')
    break