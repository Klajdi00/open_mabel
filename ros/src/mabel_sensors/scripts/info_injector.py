import rclpy, yaml, os
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from rclpy.qos import qos_profile_sensor_data # The magic QoS fix!

class CameraBridge(Node):
    def __init__(self):
        super().__init__('camera_bridge')
        
        # Define dynamic parameters
        self.declare_parameter('yaml_path', '')
        self.declare_parameter('camera_frame_id', '')
        self.declare_parameter('input_image_topic', '')
        self.declare_parameter('output_image_topic', '')
        self.declare_parameter('output_info_topic', '')

        yaml_path = os.path.expanduser(self.get_parameter('yaml_path').value)
        frame_id = self.get_parameter('camera_frame_id').value
        
        # Load the Math
        self.info_msg = CameraInfo()
        try:
            with open(yaml_path, 'r') as f:
                d = yaml.safe_load(f)
                self.info_msg.width = int(d['image_width'])
                self.info_msg.height = int(d['image_height'])
                self.info_msg.distortion_model = str(d['distortion_model'])
                self.info_msg.d = [float(x) for x in d['distortion_coefficients']['data']]
                self.info_msg.k = [float(x) for x in d['camera_matrix']['data']]
                self.info_msg.r = [float(x) for x in d['rectification_matrix']['data']]
                self.info_msg.p = [float(x) for x in d['projection_matrix']['data']]
                self.info_msg.header.frame_id = str(frame_id)
                self.get_logger().info(f"[SUCCESS] YAML loaded perfectly for {frame_id}!")
        except Exception as e:
            self.get_logger().error(f"[FATAL] Could not load YAML: {e}")

        # Setup Air-Gap Bridge Publishers
        out_img = self.get_parameter('output_image_topic').value
        out_info = self.get_parameter('output_info_topic').value
        self.pub_img = self.create_publisher(Image, out_img, qos_profile_sensor_data)
        self.pub_info = self.create_publisher(CameraInfo, out_info, qos_profile_sensor_data)
        
        # Setup Subscriber to Quarantine Namespace
        in_img = self.get_parameter('input_image_topic').value
        self.sub = self.create_subscription(Image, in_img, self.image_cb, qos_profile_sensor_data)

    def image_cb(self, msg):
        # 1. Pass the image through the bridge
        msg.header.frame_id = self.info_msg.header.frame_id
        self.pub_img.publish(msg)
        
        # 2. Attach the math and publish it simultaneously 
        self.info_msg.header.stamp = msg.header.stamp
        self.pub_info.publish(self.info_msg)

def main():
    rclpy.init()
    rclpy.spin(CameraBridge())

if __name__ == '__main__':
    main()