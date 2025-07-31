#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from geometry_msgs.msg import PoseStamped
from mavros_msgs.msg import State
from mavros_msgs.srv import CommandBool, SetMode

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

        # Create position publisher
        self.local_pos_pub = self.create_publisher(
            PoseStamped,
            'mavros/setpoint_position/local',
            10
        )

        # Create service clients
        self.arming_client = self.create_client(CommandBool, 'mavros/cmd/arming')
        self.set_mode_client = self.create_client(SetMode, 'mavros/set_mode')

        # Node variables
        self.current_state = State()
        self.pose = PoseStamped()
        self.pose.pose.position.x = 0.0
        self.pose.pose.position.y = 0.0
        self.pose.pose.position.z = 2.0
        
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

    def timer_callback(self):
        # We must be sending setpoints before switching to OFFBOARD mode
        if self.current_state.mode != 'OFFBOARD':
            # Publish the setpoint
            self.pose.header.stamp = self.get_clock().now().to_msg()
            self.local_pos_pub.publish(self.pose)

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

        # Keep publishing the setpoint
        self.pose.header.stamp = self.get_clock().now().to_msg()
        self.local_pos_pub.publish(self.pose)

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