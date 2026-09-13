import rclpy
from rclpy.node import Node
from std_msgs.msg import String


class YoloFlightControlNode(Node):
    def __init__(self):
        super().__init__('yolo_flight_control_node')

        # Remembers the last received flight mode command to avoid processing duplicates
        self.current_mode = None

        # Subscribe to the balloon tracking commands published by perception_yolo_node.py
        self.command_sub = self.create_subscription(
            String, 
            'balloon_tracking_command', 
            self.command_callback, 
            10
        )

        self.get_logger().info('YOLO flight control subscriber node is ready.')

    def command_callback(self, msg):
        requested_mode = msg.data  # Expected: "RTL" or "AUTO"
        
        # If the requested mode matches the active state, ignore duplicate triggers
        if requested_mode == self.current_mode:
            return

        self.current_mode = requested_mode
        self.get_logger().info(f"Target flight mode updated to: {requested_mode}")


def main(args=None):
    rclpy.init(args=args)
    node = YoloFlightControlNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()