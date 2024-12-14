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
    unpacked_data = struct.unpack(f"{num_points * 2}f", data)
    angles = unpacked_data[::2]  # Extract every second float as angle
    distances = unpacked_data[1::2]  # Extract every second float as distance
    return angles, distances

def run_server():
    """Receive and visualize LiDAR data over UDP."""
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    server_socket.bind((HOST, PORT))
    print(f"Server listening on {HOST}:{PORT}")

    # Matplotlib setup
    fig, ax = plt.subplots()
    sc = ax.scatter([], [], s=1)  # Point size
    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)

    def update(frame):
        try:
            data, addr = server_socket.recvfrom(8192)  # Increase buffer size to 8KB
            if data:
                print(f"Received data from {addr}, size: {len(data)}")
            else:
                print("No data received.")
                return

            # Unpack and process the data
            angles, distances = unpack_lidar_data(data)
            print(f"Angles: {angles[:5]} Distances: {distances[:5]}")  # Print first 5 values
            
            if len(angles) > 0 and len(distances) > 0:
                # Convert polar to Cartesian coordinates
                x = distances * np.cos(np.radians(angles))
                y = distances * np.sin(np.radians(angles))
                sc.set_offsets(np.c_[x, y])
            else:
                print("No valid data to visualize.")
        except Exception as e:
            print(f"Error in update function: {e}")

    ani = FuncAnimation(fig, update, interval=100)
    plt.show()

if __name__ == "__main__":
    run_server()