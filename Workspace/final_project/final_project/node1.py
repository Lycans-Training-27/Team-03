#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from mavros_msgs.msg import State
from mavros_msgs.srv import SetMode
from geometry_msgs.msg import PoseStamped
from rclpy.qos import QoSProfile, ReliabilityPolicy


class MyCustomNode(Node):
    def __init__(self):
        super().__init__("node1")

        self.current_state = State()
        self.current_pose = PoseStamped()

        self.target_x = 0.0
        self.target_y = 0.0
        self.target_z = 0.0
        self.has_target = False

        self.state_sub = self.create_subscription(State,"/mavros/state",self.state_callback,10)

        qos_profile = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,depth=10)

        self.pose_sub = self.create_subscription(PoseStamped,"/mavros/local_position/pose",self.pose_callback,qos_profile)

        self.movement_sub = self.create_subscription(String,"/flight_movement_command",self.movement_callback,10)

        self.command_sub = self.create_subscription(String,"/flight_mode_command",self.command_callback,10)

        self.set_mode_client = self.create_client(SetMode,"/mavros/set_mode")

        self.target_pub = self.create_publisher(PoseStamped,"/mavros/setpoint_position/local",10)

        self.timer = self.create_timer(0.1,self.publish_setpoint_loop)

        self.get_logger().info("node1 initialized and listening to MAVROS.")

    def state_callback(self, msg):
        if self.current_state.mode != msg.mode:
            self.get_logger().info(f"Vehicle mode changed to: {msg.mode}")

        self.current_state = msg

    def pose_callback(self, msg):
        self.current_pose = msg

    def movement_callback(self, msg):
        self.command_callback(msg)

    def command_callback(self, msg):
        command = msg.data.strip().upper()

        self.get_logger().info(f"Received command: {command}")

        if command == "MOVE RIGHT":
            self.target_x = self.current_pose.pose.position.x + 10.0
            self.target_y = self.current_pose.pose.position.y
            self.target_z = self.current_pose.pose.position.z
            self.has_target = True
            self.set_flight_mode("GUIDED")

        elif command == "MOVE LEFT":
            self.target_x = self.current_pose.pose.position.x - 10.0
            self.target_y = self.current_pose.pose.position.y
            self.target_z = self.current_pose.pose.position.z
            self.has_target = True
            self.set_flight_mode("GUIDED")

        elif command == "MOVE DOWN":
            self.target_x = self.current_pose.pose.position.x
            self.target_y = self.current_pose.pose.position.y
            self.target_z = self.current_pose.pose.position.z - 10.0
            self.has_target = True
            self.set_flight_mode("GUIDED")

        elif command in ["CONTINUE", "AUTO"]:
            self.has_target = False
            self.set_flight_mode("AUTO")
            self.get_logger().info("Path clear! Resuming original mission route.")

        elif command == "LOITER":
            self.has_target = False
            self.set_flight_mode("LOITER")

        elif command == "MANUAL":
            self.has_target = False
            self.set_flight_mode("MANUAL")

    def publish_setpoint_loop(self):
        if self.has_target and self.current_state.mode == "GUIDED":
            target = PoseStamped()

            target.header.stamp = self.get_clock().now().to_msg()
            target.header.frame_id = "map"

            target.pose.position.x = self.target_x
            target.pose.position.y = self.target_y
            target.pose.position.z = self.target_z

            self.target_pub.publish(target)

    def set_flight_mode(self, custom_mode):
        if self.current_state.mode == custom_mode:
            return

        request = SetMode.Request()
        request.custom_mode = custom_mode

        self.set_mode_client.call_async(request)

        self.get_logger().info(f"Requested mode change to: {custom_mode}")


def main(args=None):
    rclpy.init(args=args)

    node = MyCustomNode()
    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
