import socket
import numpy as np
import threading
from rplidar import RPLidar
import zlib

# RPLidar Configuration
PORT_NAME = '/dev/rplidar'  # Replace with the actual device path
BAUDRATE = 115200

# UDP Configuration
SERVER_IP = '192.168.0.6'  # Replace with the Windows server's IP address
SERVER_PORT = 5001           # UDP port number

def compress_lidar_data(scan):
    """
    Compress LiDAR data into a binary format.
    Args:
        scan: List of (quality, angle, distance) tuples.
    Returns:
        Compressed binary data.
    """
    data = np.array([[angle, distance] for _, angle, distance in scan], dtype=np.float32)
    return zlib.compress(data.tobytes())

def send_lidar_data():
    """
    Operate RPLidar, send data over UDP, and ensure proper shutdown on exit.
    """
    # Initialize RPLidar
    lidar = RPLidar(PORT_NAME, baudrate=BAUDRATE)
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    server_address = (SERVER_IP, SERVER_PORT)

    print("RPLidar connected. Sending data to server...")
    try:
        # Iterate over LiDAR scans and send data
        for scan in lidar.iter_scans():
            compressed_data = compress_lidar_data(scan)
            client_socket.sendto(compressed_data, server_address)
            print(f"Sent {len(compressed_data)} bytes to {SERVER_IP}:{SERVER_PORT}")
    except KeyboardInterrupt:
        # Handle user interruption (Ctrl+C)
        print("\nOperation interrupted by user.")
    except Exception as e:
        # Handle other exceptions
        print(f"An error occurred: {e}")
    finally:
        # Stop the LiDAR and close the socket
        print("Shutting down RPLidar and closing socket...")
        lidar.stop()
        lidar.disconnect()
        client_socket.close()
        print("Shutdown complete.")

if __name__ == "__main__":
    send_lidar_data()
