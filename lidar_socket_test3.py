import socket
import struct
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np

# UDP Configuration
HOST = '0.0.0.0'  # Listen on all interfaces
PORT = 5001       # Port number (UDP)

def unpack_lidar_data(data):
    """
    Decompress received LiDAR data.
    Args:
        data: Compressed byte data.
    Returns:
        (angles, distances): Lists of angles and distances.
    """
    num_points = len(data) // 4 // 2  # 2 floats (angle, distance) per point
    unpacked_data = np.frombuffer(data, dtype=np.float32).reshape(-1, 2)
    angles = unpacked_data[:, 0]  # Extract angles
    distances = unpacked_data[:, 1]  # Extract distances
    return angles, distances

def run_server():
    """Receive and visualize LiDAR data over UDP."""
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    server_socket.bind((HOST, PORT))
    print(f"Server listening on {HOST}:{PORT}")

    # Matplotlib setup
    fig, ax = plt.subplots()
    sc = ax.scatter([], [], s=1)  # Scatter plot for points
    ax.set_xlim(-1000, 1000)  # Set initial limits
    ax.set_ylim(-1000, 1000)

    # Preallocate arrays to avoid dynamic resizing
    max_points = 2000  # Max expected points per scan
    x = np.zeros(max_points, dtype=np.float32)
    y = np.zeros(max_points, dtype=np.float32)

    def update(frame):
        try:
            # Receive data from the client
            data, addr = server_socket.recvfrom(65536)  # Increase buffer size to 64KB
            if data:
                print(f"Received {len(data)} bytes from {addr}")

            # Unpack the received data
            angles, distances = unpack_lidar_data(data)

            # Convert polar to Cartesian coordinates
            valid_points = len(angles)  # Number of valid points
            x[:valid_points] = distances * np.cos(np.radians(angles))
            y[:valid_points] = distances * np.sin(np.radians(angles))

            # Update scatter plot
            sc.set_offsets(np.c_[x[:valid_points], y[:valid_points]])

        except Exception as e:
            print(f"Error in update function: {e}")

    ani = FuncAnimation(fig, update, interval=50)  # Reduce interval for faster updates
    plt.show()

if __name__ == "__main__":
    run_server()