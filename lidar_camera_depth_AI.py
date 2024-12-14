from collections import defaultdict
from ultralytics import YOLO
import socket
import numpy as np
import cv2
import os
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

# YOLO model initialization with GPU acceleration
model = YOLO('best.pt').to('cuda')  # Load YOLO model
names = model.names

# Directory setup for captures
crop_dir_name = "capture-picture"
crop_dir_video = "capture-video"
os.makedirs(crop_dir_name, exist_ok=True)
os.makedirs(crop_dir_video, exist_ok=True)

# Variables for tracking and processing
camera_frame = None
lidar_data = None
track_history = defaultdict(lambda: [])
frame_counter = 0
video_writer = None
idx_human = 0
idx_knife = 0
interested_object_id = None  # Target object ID
camera_lock = threading.Lock()
lidar_lock = threading.Lock()

def reset_folder(folder):
    if os.path.exists(folder):
        for file in os.listdir(folder):
            file_path = os.path.join(folder, file)
            if os.path.isfile(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                os.rmdir(file_path)
    else:
        os.mkdir(folder)

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

def process_yolo(im0):
    """Process YOLO detections and display results."""
    global frame_counter, interested_object_id, video_writer, idx_human, idx_knife

    # Run YOLO model
    results = model.track(im0, persist=True, classes=[0, 1, 3, 4, 5])  # Filter specific classes
    boxes = results[0].boxes.xyxy.cpu().numpy() if results[0].boxes.id is not None else []
    clss = results[0].boxes.cls.cpu().numpy() if results[0].boxes.id is not None else []
    track_ids = results[0].boxes.id.int().cpu().tolist() if results[0].boxes.id is not None else []

    # Highlight detected objects
    if boxes is not None:
        for box, cls, obj_id in zip(boxes, clss, track_ids):
            x1, y1, x2, y2 = map(int, box)
            label = f"ID {obj_id}, Class {names[int(cls)]}"
            cv2.rectangle(im0, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(im0, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    return im0

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
            processed_frame = process_yolo(camera_frame.copy())  # Apply YOLO processing
            camera_image.set_data(cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB))  # Update the plot

if __name__ == "__main__":
    # Start LiDAR and camera server threads
    lidar_thread = threading.Thread(target=lidar_server, daemon=True)
    camera_thread = threading.Thread(target=camera_server, daemon=True)

    lidar_thread.start()
    camera_thread.start()

    # Start visualization
    ani = FuncAnimation(fig, update, interval=50)  # 50ms interval for updates
    plt.show()