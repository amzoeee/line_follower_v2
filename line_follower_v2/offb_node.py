#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from mavros_msgs.msg import State, PositionTarget
from mavros_msgs.srv import CommandBool, SetMode
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import Image
import numpy as np
import cv2
from tf_transformations import euler_from_quaternion
import math


###########
#CONSTANTS#
###########
MAX_YAW_SPEED = 5.0 # rad per second, counterclockwise
MAX_X_SPEED = 1.0 # meters per second, forward
MAX_Y_SPEED = 1.0 # meters per second, right
MAX_Z_SPEED = 1.0 # meters per second, down

TAKEOFF_ALTITUDE = 1.0 # meters
TAKEOFF_TIME = 10 # seconds

IMAGE_WIDTH, IMAGE_HEIGHT = 1280, 960 # pixels

CENTER = np.array([IMAGE_WIDTH//2, IMAGE_HEIGHT//2]) # Center of the image frame. We will treat this as the center of mass of the drone
EXTEND = 300 # Number of pixels forward to extrapolate the line

#PID Constants
KP_X = 0.0
KP_Y = 0.0
KP_W_Z = 0.0

KD_X = 0.0
KD_Y = 0.0
KD_W_Z = 0.0

prev_x_error = 0
prev_y_error = 0
prev_angle_error = 0

LOW = np.array([250, 250, 250])  # Lower image thresholding bound
HI = np.array([255, 255, 255])   # Upper image thresholding bound

KERNEL_D = np.ones((30, 30), np.uint8)
KERNEL_E = np.ones((20, 20), np.uint8)

R_dc2bd = np.array([[0.0, -1.0, 0.0, 0.0], 
                      [1.0, 0.0, 0.0, 0.0], 
                      [0.0, 0.0, 1.0, 0.0], 
                      [0.0, 0.0, 0.0, 1.0]]) 

# --- Bitmask Constants ---
# (see http://docs.ros.org/en/noetic/api/mavros_msgs/html/msg/PositionTarget.html)
IGNORE_PX = 1
IGNORE_PY = 2
IGNORE_PZ = 4
IGNORE_VX = 8
IGNORE_VY = 16
IGNORE_VZ = 32
IGNORE_AFX = 64
IGNORE_AFY = 128
IGNORE_AFZ = 256
IGNORE_YAW = 1024
IGNORE_YAW_RATE = 2048

class OffboardControlNode(Node):

    def __init__(self):
        super().__init__('offb_node_py')

        # Configure QoS Profiles for communication with MAVROS
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )
        
        # Create state subscriber
        self.state_sub = self.create_subscription(
            State,
            'mavros/state',
            self.state_cb,
            qos_profile
        )

        self.pose_sub = self.create_subscription(
            PoseStamped,
            '/mavros/local_position/pose',
            self.pose_cb,
            10
        )

        self.camera_sub = self.create_subscription(
            Image,
            '/world/line_following_track/model/x500_mono_cam_down_0/link/camera_link/sensor/imager/image', # for sim 
            # '/camera_1/image_raw', # for real
            self.camera_cb,
            qos_profile
        )

        # Create pos/vel publisher
        self.local_setpoint_pub = self.create_publisher(
            PositionTarget,
            'mavros/setpoint_raw/local',
            10
        )

        self.detector_image_pub = self.create_publisher(Image, '/line/detector_image', 1)

        # Create service clients
        self.arming_client = self.create_client(CommandBool, 'mavros/cmd/arming')
        self.set_mode_client = self.create_client(SetMode, 'mavros/set_mode')

        # Node variables
        self.current_state = State()
        self.current_pose = PoseStamped()
        self.setpoint = PositionTarget()
        self.setpoint.coordinate_frame = PositionTarget.FRAME_LOCAL_NED
        self.setpoint.type_mask = IGNORE_PX | IGNORE_PY | IGNORE_VZ | IGNORE_AFX | IGNORE_AFY | IGNORE_AFZ | IGNORE_YAW_RATE
        
        # Wait for MAVROS services to be available
        while not self.arming_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for /mavros/cmd/arming service...')
        while not self.set_mode_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for /mavros/set_mode service...')

        self.get_logger().info("MAVROS services available.")
        
        # Create a timer to run the main loop
        self.timer = self.create_timer(0.05, self.timer_callback) # 20 Hz

    def state_cb(self, msg):
        self.current_state = msg

    def pose_cb(self, msg):
        self.current_pose = msg

    def camera_cb(self, msg):
        # Convert Image msg to OpenCV image
        image = self.bridge.imgmsg_to_cv2(msg, "mono8")

        # Detect line in the image. detect returns a parameterize the line (if one exists)
        line = self.detect_line(image)

        # If a line was detected, publish the parameterization to the topic '/line/param'
        if line is not None:
            self.x, self.y, self.vx, self.vy = line

            # Publish annotated image 
            # Draw the detected line on a color version of the image
            annotated = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            x, y, vx, vy = line
            pt1 = (int(x - 100*vx), int(y - 100*vy))
            pt2 = (int(x + 100*vx), int(y + 100*vy))
            # pt1 = int(x), int(y)
            # pt2 = int(vx), int(vy)
            cv2.line(annotated, pt1, pt2, (0, 0, 255), 2)
            cv2.circle(annotated, (int(x), int(y)), 5, (0, 255, 0), -1)
            # Convert to ROS Image message and publish
            annotated_msg = self.bridge.cv2_to_imgmsg(annotated, "bgr8")
            self.detector_image_pub.publish(annotated_msg)

    def detect_line(self, image):
        """ 
        Given an image, fit a line to biggest contour if it meets size requirements (otherwise return None)
        and return a parameterization of the line as a center point on the line and a vector
        pointing in the direction of the line.
            Args:
                - image = OpenCV image
            Returns: (x, y, vx, vy) where (x, y) is the centerpoint of the line in image and 
            (vx, vy) is a vector pointing in the direction of the line. Both values are given
            in downward camera pixel coordinates. Returns None if no line is found
        """
        
        h, w = image.shape
        
        kernel_size = 30
        kernel = np.ones((kernel_size,kernel_size), np.uint8) 

        kernel_size = 20
        kernel2 = np.ones((kernel_size,kernel_size), np.uint8)

        # dilate + erode 
        image = cv2.dilate(image, kernel,iterations = 1)
        image = cv2.erode(image, kernel2,iterations = 1)

        _, threshold = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY)

        contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        cnt_sort = lambda cnt: (max(cv2.minAreaRect(cnt)[1])) # sort by largest height/width 

        sorted_contours = sorted(contours, key=cnt_sort, reverse=True)

        if len(sorted_contours) > 0:
        
            all_points = np.vstack(sorted_contours[0])
            [vx, vy, x, y] = cv2.fitLine(all_points, cv2.DIST_L2, 0, 0.01, 0.01)
                    
            return float(x), float(y), float(vx), float(vy)

    def run_drone(self):
        x, y, vx, vy = self.x, self.y, self.vx, self.vy

        line_point = np.array([x, y])
        line_dir = np.array([vx, vy])
        line_dir = line_dir / np.linalg.norm(line_dir)  # Ensure unit vector

        if line_dir[1] < 0:
            line_dir = -line_dir

        # Target point EXTEND pixels ahead along the line direction
        target = line_point + EXTEND * line_dir

        # Error between center and target
        error = target - CENTER

        # Set linear velocities (downward camera frame)
        self.vx__dc = KP_X * error[0]
        self.vy__dc = KP_Y * error[1]

        self.vx__dc += KD_X * (error[0]-self.prev_x_error)/0.1
        self.vy__dc += KD_Y * (error[1]-self.prev_y_error)/0.1

        self.prev_x_error = error[0]
        self.prev_y_error = error[1]

        # Get angle between y-axis and line direction
        # Positive angle is counter-clockwise
        forward = np.array([0.0, 1.0])
        angle_error = math.atan2(-line_dir[0], line_dir[1])

        # Set angular velocity (yaw)
        self.wz__dc = KP_W_Z * angle_error

        self.wz__dc += KD_W_Z * (angle_error-self.prev_w_error)/0.1

        self.prev_w_error = angle_error

        self.update_setpoint(self.convert_velocity_setpoints())

        self.get_logger().info(f"x error: {error[0]}, y error: {error[1]}, angle error: {angle_error}")

    def update_setpoint(self, setpoint):
        self.setpoint.velocity.x = setpoint[0]
        self.setpoint.velocity.y = setpoint[1]
        self.setpoint.yaw_rate = setpoint[2]
        self.setpoint.position.z = TAKEOFF_ALTITUDE

    def convert_velocity_setpoints(self):
        '''Convert velocity setpoints from downward camera frame to lenu frame'''
        vx, vy, vz = self.dc2lned((self.vx__dc, self.vy__dc, self.vz__dc))
        _, _, wz = self.dc2lned((0.0, 0.0, self.wz__dc))

        vx = min(max(vx,-MAX_X_SPEED), MAX_X_SPEED)
        vy = min(max(vy,-MAX_Y_SPEED), MAX_Y_SPEED)
        wz = min(max(wz,-MAX_YAW_SPEED), MAX_YAW_SPEED)

        return (vx, vy, wz)

    def dc2lned(self, vector):
        '''Use current yaw to convert vector from downward camera frame to lned frame'''
        v4 = np.array([[vector[0]],
                        [vector[1]],
                        [vector[2]],
                        [     0.0]])
        
        
        quaternion = self.current_pose.orientation
        roll, pitch, yaw = euler_from_quaternion(quaternion)

        self.get_logger().info("yaw: " + str(yaw))

        R_dc2bd = np.array([[0.0, -1.0, 0.0, 0.0]
                            [1.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 1.0, 0.0],
                            [0.0, 0.0, 0.0, 1.0]])

        R_dc2lned = np.array([[-np.sin(yaw), np.cos(yaw), 0.0, 0.0], 
                                 [np.cos(yaw), np.sin(yaw), 0.0, 0.0], 
                                 [0.0, 0.0, 1.0, 0.0], 
                                 [0.0, 0.0, 0.0, 1.0]]) 

        output = np.dot(np.dot(R_dc2bd, R_dc2lned), v4)
        
        return (output[0,0], output[1,0], output[2,0])
        
    

    def timer_callback(self):
        # We must be sending setpoints before switching to OFFBOARD mode
        if self.current_state.mode != 'OFFBOARD':
            # Publish the setpoint
            self.setpoint.header.stamp = self.get_clock().now().to_msg()
            self.setpoint.velocity.x = 0.0
            self.setpoint.velocity.y = 0.0
            self.setpoint.yaw_rate = 0.0
            self.setpoint.position.z = TAKEOFF_ALTITUDE
            self.local_setpoint_pub.publish(self.setpoint)

            # Try to switch to OFFBOARD mode
            if self.current_state.connected:
                set_mode_req = SetMode.Request()
                set_mode_req.custom_mode = 'OFFBOARD'
                self.set_mode_client.call_async(set_mode_req)

        # Once in OFFBOARD mode, we can arm the vehicle
        if self.current_state.mode == 'OFFBOARD' and not self.current_state.armed:
            if self.current_state.connected:
                arm_cmd_req = CommandBool.Request()
                arm_cmd_req.value = True
                self.arming_client.call_async(arm_cmd_req)
                self.get_logger().info("Arming commanded...", once=True)

        if self.current_state.armed:
            self.get_logger().info("Armed.", once=True)

            self.run_drone()

        # Keep publishing the setpoint
        self.setpoint.header.stamp = self.get_clock().now().to_msg()
        self.local_setpoint_pub.publish(self.setpoint)

def main(args=None):
    rclpy.init(args=args)
    node = OffboardControlNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Node interrupted by user.')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()