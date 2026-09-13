import cv2
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from ultralytics import YOLO


CAMERA_SOURCE = 'rtsp://192.168.1.42:554/stream'

MODEL_PATH = "best.pt"  # trained YOLO weights file, must be in the same folder or given as a full path
CONFIDENCE_THRESHOLD = 0.5  # ignore detections YOLO itself isn't at least 50% confident about

LOOP_RATE_HZ = 15.0  # targets a 15 Hz execution speed, same as the main pipeline

CONFIRM_NEEDED = 2  # needs 2 consecutive frames in the same sector before trusting it, to avoid single-frame glitches
CLEAR_NEEDED = 3    # needs 3 consecutive frames with no valid sector before clearing back to no-command


class PerceptionYoloNode(Node):
    def __init__(self):
        super().__init__('perception_yolo_node')

        # ROS 2 Publisher -- single topic for this bonus-task node
        self.command_pub = self.create_publisher(String, 'balloon_tracking_command', 10)

        # load the trained YOLO model once here at startup, not inside the loop, since loading it is slow
        self.get_logger().info(f"Loading YOLO model from: {MODEL_PATH}")
        self.model = YOLO(MODEL_PATH)

        # Camera Setup -- tries the phone stream first, falls back to a local webcam
        self.cap = cv2.VideoCapture(CAMERA_SOURCE, cv2.CAP_FFMPEG)  # open the video capture object using the FFmpeg backend
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # reduce buffer size to 1 frame to minimize video latency

        # Fallback to local USB webcam (0) if the phone stream is unavailable
        if not self.cap.isOpened():
            self.get_logger().warn(f"Phone stream {CAMERA_SOURCE} unavailable. Trying default webcam (0)...")
            self.cap = cv2.VideoCapture(0)

        # verify camera status
        if not self.cap.isOpened():
            self.get_logger().error("No camera source found! Connect a webcam or start the phone stream.")
        else:
            self.get_logger().info("Camera/video source initialized successfully.")

        # counters used to avoid reacting to a single lucky/unlucky frame
        self.seen_count = 0
        self.miss_count = 0
        self.confirmed_sector = None  # the sector we currently trust, or None if nothing confirmed

        # remembers the last command we actually sent, so we only publish when it changes
        self.last_published_cmd = None

        # set up a ROS 2 timer loop running at 15 Hz that automatically calls process_frame()
        self.timer = self.create_timer(1.0 / LOOP_RATE_HZ, self.process_frame)

    # read camera frame, run YOLO detection, figure out grid sector, and publish a command
    def process_frame(self):
        # flush a few stale buffered frames first, then read the newest one (same latency fix as the main pipeline)
        for _ in range(4):
            if not self.cap.grab():
                break
        ret, frame = self.cap.retrieve()
        if not ret:
            self.get_logger().warn("Failed to grab frame from video source", throttle_duration_sec=3.0)  # verifies a valid frame was captured; logs a throttled warning and exits early to avoid crashing the node
            return

        h, w, _ = frame.shape

        # run the YOLO model on this single frame -- returns a list of detections, each with a box and a confidence score
        results = self.model.predict(frame, conf=CONFIDENCE_THRESHOLD, verbose=False)
        boxes = results[0].boxes  # all detected boxes for this frame

        # reset detection variables for this frame
        detected_sector = None
        best_conf = 0.0
        best_x1 = best_y1 = best_x2 = best_y2 = None

        # manually loop through every detected box to find the one with the highest confidence
        # (done as a plain loop instead of a shortcut function, so it's easy to follow)
        num_boxes = len(boxes)
        i = 0
        while i < num_boxes:
            conf = float(boxes.conf[i])  # confidence score for this specific detection
            if conf > best_conf:
                best_conf = conf
                coords = boxes.xyxy[i]  # bounding box as [x1, y1, x2, y2] pixel coordinates
                best_x1 = float(coords[0])
                best_y1 = float(coords[1])
                best_x2 = float(coords[2])
                best_y2 = float(coords[3])
            i = i + 1

        # if we found at least one detection, figure out which grid sector its center falls into
        if best_x1 is not None:
            center_x = (best_x1 + best_x2) / 2.0  # horizontal center of the bounding box
            center_y = (best_y1 + best_y2) / 2.0  # vertical center of the bounding box

            # figure out the column: 0 = left third, 1 = middle third, 2 = right third
            if center_x < w / 3:
                column = 0
            elif center_x < (2 * w / 3):
                column = 1
            else:
                column = 2

            # figure out the row: 0 = top third, 1 = middle third, 2 = bottom third
            if center_y < h / 3:
                row = 0
            elif center_y < (2 * h / 3):
                row = 1
            else:
                row = 2

            # only the four CORNER sectors matter for this task, per the spec
            if row == 0 and column == 0:
                detected_sector = 'top_left'
            elif row == 0 and column == 2:
                detected_sector = 'top_right'
            elif row == 2 and column == 0:
                detected_sector = 'bottom_left'
            elif row == 2 and column == 2:
                detected_sector = 'bottom_right'
            # any other sector (middle row, middle column, or center) leaves detected_sector as None

            # draw the bounding box around the detected balloon
            cv2.rectangle(frame, (int(best_x1), int(best_y1)), (int(best_x2), int(best_y2)), (0, 255, 255), 2)

        # updates detection counts to ensure the sector is present consistently before trusting it
        if detected_sector is not None:
            self.seen_count = self.seen_count + 1
            self.miss_count = 0
        else:
            self.miss_count = self.miss_count + 1
            self.seen_count = 0

        # only trust a sector once we've seen it consistently, and only clear it after several clean frames
        if self.seen_count >= CONFIRM_NEEDED:
            self.confirmed_sector = detected_sector
        if self.miss_count >= CLEAR_NEEDED:
            self.confirmed_sector = None

        # map the confirmed sector to the actual flight command
        command = None
        if self.confirmed_sector == 'top_left' or self.confirmed_sector == 'top_right':
            command = 'RTL'
        elif self.confirmed_sector == 'bottom_left' or self.confirmed_sector == 'bottom_right':
            command = 'AUTO'
        # any other confirmed_sector value (or None) leaves command as None, nothing gets published

        # PUBLISH COMMAND (Publish-on-Change)
        # publish only if we have a real command AND it's different from the last one we sent
        if command is not None and command != self.last_published_cmd:
            msg = String()
            msg.data = command
            self.command_pub.publish(msg)
            self.get_logger().info(f"[Balloon] Published: '{command}' (sector: {self.confirmed_sector})")
            self.last_published_cmd = command

        # if there's no command right now, reset last_published_cmd so the next real command gets sent again
        if command is None:
            self.last_published_cmd = None

        # draw the 3x3 grid lines on screen so it's easy to see the sectors visually
        cv2.line(frame, (int(w / 3), 0), (int(w / 3), h), (100, 100, 100), 1)
        cv2.line(frame, (int(2 * w / 3), 0), (int(2 * w / 3), h), (100, 100, 100), 1)
        cv2.line(frame, (0, int(h / 3)), (w, int(h / 3)), (100, 100, 100), 1)
        cv2.line(frame, (0, int(2 * h / 3)), (w, int(2 * h / 3)), (100, 100, 100), 1)

        # shows what the camera sees + grid + status text
        status_text = "Sector: " + str(self.confirmed_sector) + "  Command: " + str(command)
        cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.imshow("YOLO Balloon Tracking", frame)
        cv2.waitKey(1)


def main(args=None):
    rclpy.init(args=args)
    node = PerceptionYoloNode()
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