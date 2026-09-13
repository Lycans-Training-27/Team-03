import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import String


CAMERA_SOURCE = 'rtsp://'
LOOP_RATE_HZ = 15.0 # targets a 15 Hz execution speed

# Define HSV colour thresholds
ORANGE_LOWER, ORANGE_UPPER = np.array([5, 130, 130]), np.array([20, 255, 255])
GREEN_LOWER, GREEN_UPPER = np.array([40, 120, 80]), np.array([80, 255, 255])
BLUE_LOWER, BLUE_UPPER = np.array([95, 120, 80]), np.array([135, 255, 255])
 
CONFIRM_NEEDED = 2  # it needs 2 consecutive frames to trigger an avoidance command to avoid single_frame glitches
CLEAR_NEEDED = 3 # once an obstacle is confirmed, It must be completely absent for 3 video frames in a row before the system clears the hazard alert

PROXIMITY_HEIGHT_FRAC = 0.15  # Check if the obstacle is close enough by verifying it takes up at least 15% of the frame height


class PerceptionPublisherNode(Node):
    def __init__(self):
        super().__init__('perception_publisher_node')

        # ROS 2 Publishers
        self.obstacle_pub = self.create_publisher(String, 'flight_movement_command', 10)
        self.target_pub = self.create_publisher(String, 'flight_mode_command', 10)

        # Camera Setup
        src = int(CAMERA_SOURCE) if str(CAMERA_SOURCE).isdigit() else CAMERA_SOURCE # automatically convert camera index to an integer for webcams or keep as a string for stream URLs
        self.cap = cv2.VideoCapture(src, cv2.CAP_FFMPEG)  # open the video capture object using the FFmpeg backend
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # reduce buffer size to 1 frame to minimize video latency

        # verify camera status
        if not self.cap.isOpened():
            self.get_logger().error(f"Could not open camera/source: {CAMERA_SOURCE}")
        else:
            self.get_logger().info(f"Successfully connected to camera source: {CAMERA_SOURCE}")

        # tracking how many consecutive frames an obstacle has been detected or missed, and stores whether an obstacle is confirmed
        self.hazard_seen_count = 0
        self.hazard_miss_count = 0
        self.hazard_confirmed = False

        # tracking consecutive detection counts, miss counts, and confirmation statuses for each flight mode target (RTL and LAND) independently
        self.target_seen_count = {"LOITER": 0, "MANUAL": 0}
        self.target_miss_count = {"LOITER": 0, "MANUAL": 0}
        self.target_confirmed = {"LOITER": False, "MANUAL": False}

        # tracking previous states to only publish on change
        self.last_obstacle_cmd = None
        self.last_target_mode = None

        # set up a ROS 2 timer loop running at 15 Hz that automatically calls process_frame()
        self.timer = self.create_timer(1.0 / LOOP_RATE_HZ, self.process_frame)

    # read camera frame, extract dimensions, and convert to HSV color space
    def process_frame(self):
        for _ in range(4):
            if not self.cap.grab():
             break
        ret, frame = self.cap.retrieve()
        if not ret:
            self.get_logger().warn("Failed to grab frame from video source!", throttle_duration_sec=3.0)
            return
        h, w, _ = frame.shape
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        mask_orange = cv2.inRange(hsv, ORANGE_LOWER, ORANGE_UPPER) # creates a binary black-and-white mask where orange pixels become white (255) and all other colors become black (0).
        contours, _ = cv2.findContours(mask_orange, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # extract external shape contours from the binary mask


        # Reset frame detection flag and horizontal center coordinate
        seen_this_frame = False
        cx = None

        if contours:
            largest = max(contours, key=cv2.contourArea) # select the largest detected orange contour to ignore smaller noise blobs

            # verifies that the largest shape takes up at least 0.2% of total screen area, then extracts its bounding rectangle coordinates 
            if cv2.contourArea(largest) > (h * w * 0.002): 
                x, y, bw, bh = cv2.boundingRect(largest)
               
                if (bh / h) >= PROXIMITY_HEIGHT_FRAC:  # only count it if it's big enough on screen
                    seen_this_frame = True
                    cx = x + bw / 2.0 # calculate center x
                    cv2.rectangle(frame, (x, y), (x + bw, y + bh), (0, 0, 255), 2) # draw bounding box

        # updates detection counts to ensure the obstacle is present before setting hazard_confirmed to True
        if seen_this_frame:
            self.hazard_seen_count += 1
            self.hazard_miss_count = 0
        else:
            self.hazard_miss_count += 1
            self.hazard_seen_count = 0

        if self.hazard_seen_count >= CONFIRM_NEEDED:
            self.hazard_confirmed = True
        if self.hazard_miss_count >= CLEAR_NEEDED:
            self.hazard_confirmed = False

        obstacle_cmd = "CONTINUE" # Set default movement command to CONTINUE


        if self.hazard_confirmed and cx is not None:
            if cx < w * 0.4: # if the obstacle's center is on the left 40% of the screen, it commands the vehicle to dodge to the right
                obstacle_cmd = "MOVE RIGHT"
            elif cx > w * 0.6: # if the obstacle's center is on the right 40% of the screen, it commands the vehicle to dodge to the left
                obstacle_cmd = "MOVE LEFT"
            else: # If the obstacle is in the center 20% of the screen (between 40% and 60% of frame width), it commands the vehicle to move lower/downward
                obstacle_cmd = "MOVE DOWN"
        
       
        # TARGET DETECTION (Green Circle / Blue Square)

        mode_cmd = "NONE" # no target mode trigger

        # loops through both target rules: Green Circle for Return-To-Launch (RTL) and Blue Square for LAND
        for lower, upper, shape_type, mode_name, color in [(GREEN_LOWER, GREEN_UPPER, "circle", "LOITER", (0, 255, 0)),(BLUE_LOWER, BLUE_UPPER, "square", "MANUAL", (255, 0, 0))]:
            mask = cv2.inRange(hsv, lower, upper) # create binary mask for current target color range
            target_contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # find external contours for the target color mask

            seen_target = False
            # select largest contour and calculate its surface area and perimeter
            if target_contours:
                largest_target = max(target_contours, key=cv2.contourArea)
                area = cv2.contourArea(largest_target)
                peri = cv2.arcLength(largest_target, True)

                if area > (h * w * 0.003) and peri > 0:  # filter out small background noise and ensure valid perimeter length
                    # compute shape circularity and bounding box aspect ratio
                    circularity = (4 * np.pi * area) / (peri ** 2)
                    bx, by, bw, bh = cv2.boundingRect(largest_target)  # extract bounding box coordinates (x, y, width, height) around the target contour
                    aspect_ratio = bw / float(bh)

                    # verify geometric criteria (circle or square) and draw colored bounding box
                    if (shape_type == "circle" and circularity >= 0.65) or \
                       (shape_type == "square" and 0.7 <= aspect_ratio <= 1.3):
                        seen_target = True
                        cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), color, 2) # draw target bounding box overlay on the video frame using target color

            if seen_target:
                self.target_seen_count[mode_name] += 1
                self.target_miss_count[mode_name] = 0
            else:
                self.target_miss_count[mode_name] += 1
                self.target_seen_count[mode_name] = 0

            if self.target_seen_count[mode_name] >= CONFIRM_NEEDED:
                self.target_confirmed[mode_name] = True
            if self.target_miss_count[mode_name] >= CLEAR_NEEDED:
                self.target_confirmed[mode_name] = False

            if self.target_confirmed[mode_name]: # check if the target detection for this flight mode is confirmed
                mode_cmd = mode_name  # assign active flight mode command
                break # stop checking remaining target shapes once a valid target is confirmed

       
        # PUBLISH COMMANDS (Publish-on-Change)

        if obstacle_cmd != self.last_obstacle_cmd:
            msg = String()
            msg.data = obstacle_cmd
            self.obstacle_pub.publish(msg)
            self.get_logger().info(f"[Obstacle] Published: '{obstacle_cmd}'")
            self.last_obstacle_cmd = obstacle_cmd

        if mode_cmd != "NONE" and mode_cmd != self.last_target_mode:
            msg = String()
            msg.data = mode_cmd
            self.target_pub.publish(msg)
            self.get_logger().info(f"[Mode Target] Published: '{mode_cmd}'")
            self.last_target_mode = mode_cmd

       
        # shows what the camera sees + bounding boxes
        cv2.putText(frame, f"Obstacle: {obstacle_cmd}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, f"Target: {mode_cmd}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow("Perception - Live Detection", frame)
        cv2.waitKey(1)  

def main(args=None):
    rclpy.init(args=args)
    node = PerceptionPublisherNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.cap.release()
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
