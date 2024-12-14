import socket
import json
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan

# UDP Configuration
PC_IP = '192.168.0.6'  # Replace with your PC's IP
PC_PORT = 5001  # Port number

class LidarSender(Node):
    def __init__(self):
        super().__init__('lidar_sender')
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.pc_address = (PC_IP, PC_PORT)

        # Subscribe to /scan topic
        self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)
        self.get_logger().info("LidarSender node started.")

    def scan_callback(self, msg):
        """Send LiDAR data to PC."""
        lidar_data = {"angles": [], "distances": []}

        for i, distance in enumerate(msg.ranges):
            angle = msg.angle_min + i * msg.angle_increment
            lidar_data["angles"].append(angle)
            lidar_data["distances"].append(distance)

        try:
            message = json.dumps(lidar_data)
            self.sock.sendto(message.encode(), self.pc_address)
            self.get_logger().info("Sent LiDAR data to PC.")
        except Exception as e:
            self.get_logger().error(f"Error sending data: {e}")


def main(args=None):
    rclpy.init(args=args)
    node = LidarSender()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down node.")
    finally:
        node.sock.close()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
