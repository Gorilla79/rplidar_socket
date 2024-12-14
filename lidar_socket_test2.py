import socket
import threading
import numpy as np
import zlib
import pyqtgraph as pg
from pyqtgraph.Qt import QtGui

HOST = '0.0.0.0'
PORT = 5001

def unpack_lidar_data(data):
    decompressed_data = zlib.decompress(data)
    arr = np.frombuffer(decompressed_data, dtype=np.float32).reshape(-1, 2)
    return arr[:, 0], arr[:, 1]

def run_server():
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    server_socket.bind((HOST, PORT))
    print(f"Server listening on {HOST}:{PORT}")

    app = QtGui.QApplication([])
    win = pg.GraphicsLayoutWidget()
    plot = win.addPlot()
    scatter = pg.ScatterPlotItem(size=1)
    plot.addItem(scatter)
    win.show()

    def update_plot(x, y):
        scatter.setData(x, y)

    def receive_loop():
        while True:
            data, addr = server_socket.recvfrom(65536)
            angles, distances = unpack_lidar_data(data)
            x = distances * np.cos(np.radians(angles))
            y = distances * np.sin(np.radians(angles))
            update_plot(x, y)

    threading.Thread(target=receive_loop, daemon=True).start()
    app.exec_()

if __name__ == "__main__":
    run_server()