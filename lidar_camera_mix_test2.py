import socket
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import threading

# Configuration
CAMERA_HOST = '0.0.0.0'
CAMERA_PORT = 5000
LIDAR_HOST = '0.0.0.0'
LIDAR_PORT = 5001

# Visualization setup
fig, (ax_lidar, ax_camera) = plt.subplots(1, 2)
sc = ax_lidar.scatter([], [], s=1)  # Scatter plot for LiDAR points
ax_lidar.set_xlim(-3000, 3000)
ax_lidar.set_ylim(-3000, 3000)
ax_lidar.set_title("LiDAR Point Cloud")

ax_camera.set_title("Camera Stream")
ax_camera.axis('off')  # Hide axis for camera stream
camera_image = ax_camera.imshow(np.zeros((480, 640, 3), dtype=np.uint8))

# Preallocate arrays for LiDAR
max_points = 2000
x = np.zeros(max_points, dtype=np.float32)
y = np.zeros(max_points, dtype=np.float32)

# Shared data between threads
camera_frame = None
lidar_data = None
lidar_lock = threading.Lock()
camera_lock = threading.Lock()

def unpack_lidar_data(data):
    """Decompress LiDAR data."""
    unpacked_data = np.frombuffer(data, dtype=np.float32).reshape(-1, 2)
    angles = unpacked_data[:, 0]
    distances = unpacked_data[:, 1]
    return angles, distances

def lidar_server():
    """Receive and store LiDAR data."""
    global lidar_data
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    server_socket.bind((LIDAR_HOST, LIDAR_PORT))
    print(f"LiDAR server listening on {LIDAR_HOST}:{LIDAR_PORT}")
    while True:
        try:
            data, _ = server_socket.recvfrom(65536)
            with lidar_lock:
                lidar_data = unpack_lidar_data(data)
        except Exception as e:
            print(f"LiDAR server error: {e}")

def camera_server():
    """Receive and store camera frames."""
    global camera_frame
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    server_socket.bind((CAMERA_HOST, CAMERA_PORT))
    print(f"Camera server listening on {CAMERA_HOST}:{CAMERA_PORT}")
    while True:
        try:
            data, _ = server_socket.recvfrom(65536)
            np_arr = np.frombuffer(data, np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            with camera_lock:
                camera_frame = frame
        except Exception as e:
            print(f"Camera server error: {e}")

def update(frame):
    """Update the visualization."""
    global camera_frame, lidar_data

    # Update LiDAR visualization
    with lidar_lock:
        if lidar_data:
            angles, distances = lidar_data
            valid_points = len(angles)
            x[:valid_points] = distances * np.cos(np.radians(angles))
            y[:valid_points] = distances * np.sin(np.radians(angles))
            sc.set_offsets(np.c_[x[:valid_points], y[:valid_points]])

    # Update camera visualization
    with camera_lock:
        if camera_frame is not None:
            camera_image.set_data(cv2.cvtColor(camera_frame, cv2.COLOR_BGR2RGB))

if __name__ == "__main__":
    # Start LiDAR and camera server threads
    lidar_thread = threading.Thread(target=lidar_server, daemon=True)
    camera_thread = threading.Thread(target=camera_server, daemon=True)

    lidar_thread.start()
    camera_thread.start()

    # Start visualization
    ani = FuncAnimation(fig, update, interval=50)  # 50ms interval for updates
    plt.show()