import socket
from rplidar import RPLidar
import struct

# RPLidar Configuration
PORT_NAME = '/dev/rplidar'  # RPLidar device path
BAUDRATE = 115200

# UDP Configuration
SERVER_IP = '192.168.0.6'  # Replace with Windows server IP
SERVER_PORT = 5001           # Port number (UDP)

def compress_lidar_data(scan):
    """
    Compress LiDAR data into a format suitable for UDP transmission.
    Args:
        scan: (quality, angle, distance) list of tuples.
    Returns:
        Compressed byte data.
    """
    # Extract only angle and distance, ignoring quality
    compressed_data = struct.pack(
        f"{len(scan) * 2}f",
        *(value for pair in ((angle, distance) for _, angle, distance in scan) for value in pair)
    )
    return compressed_data

def send_lidar_data():
    """Send LiDAR data via UDP."""
    lidar = RPLidar(PORT_NAME, baudrate=BAUDRATE)
    print("RPLidar connected.")

    # Create UDP socket
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    server_address = (SERVER_IP, SERVER_PORT)

    try:
        for scan in lidar.iter_scans():
            compressed_data = compress_lidar_data(scan)
            client_socket.sendto(compressed_data, server_address)
    except KeyboardInterrupt:
        print("Operation interrupted.")
    finally:
        lidar.stop()
        lidar.disconnect()
        client_socket.close()
        print("LiDAR and socket connection closed.")
        
if __name__ == "__main__":
    send_lidar_data()
