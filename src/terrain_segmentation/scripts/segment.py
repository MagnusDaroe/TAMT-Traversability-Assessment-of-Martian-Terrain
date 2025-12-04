#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from ultralytics import YOLO
import os
import yaml
import cv2
import numpy as np
from ament_index_python.packages import get_package_share_directory

class TAMTSegmentationNode(Node):
    def __init__(self):
        super().__init__('tamt_segmentation_node')

        # Declare parameter for config file
        self.declare_parameter('config_file', 'seg_inference.yaml')
        config_file = self.get_parameter('config_file').value
        
        # Load configuration
        self.config = self.load_config(config_file)

        # Get inference parameters
        self.conf = self.config['inference'].get('conf', 0.75)
        self.iou = self.config['inference'].get('iou', 0.50)
        self.imgsz = self.config['inference'].get('imgsz', 640)
        self.max_det = self.config['inference'].get('max_det', 300)

        # Initialize CV bridge
        self.bridge = CvBridge()
        
        # Load model
        self.model = None
        self.load_model()

        # Header of received image
        self.current_image_header = None
                
        # ------ Publishers ------
        self.segmentation_mask_pub = self.create_publisher(Image, '/tamt/segmentation/masks_with_confidence', 10)
      
        # ------ Subscribers ------
        self.image_sub = self.create_subscription(
            Image,
            '/tamt/sync_rgb',
            self.image_callback,
            10
        )
 
        self.get_logger().info('TAMT Segmentation Node initialized')
        self.log_config()

    #################### Subscriber & Publisher functions ####################

    def image_callback(self, msg):
        # Convert ROS Image message to OpenCV format
        self.current_image_header = msg.header
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        
        # Run inference
        self.run_inference(cv_image)

    def publish_mask_with_confidence(self, mask_with_confidence):
        """Publish segmentation mask to ROS topic"""
        try:
            # Convert to ROS Image message with 16UC1 encoding
            mask_msg = self.bridge.cv2_to_imgmsg(mask_with_confidence, encoding="16UC1")
            mask_msg.header = self.current_image_header  # Set the header
            self.segmentation_mask_pub.publish(mask_msg)
            
        except Exception as e:
            self.get_logger().error(f"Failed to publish mask: {str(e)}")
    

    #################### Load config and model ####################
       
    def load_config(self, config_file):
        """Load configuration from YAML file"""
        try:
            # Try to find config in package share directory
            try:
                package_share = get_package_share_directory('terrain_segmentation')
                config_path = os.path.join(package_share, 'config', config_file)
            except:
                config_path = config_file
                
            if not os.path.exists(config_path):
                # Try absolute path
                config_path = config_file
                
            if not os.path.exists(config_path):
                self.get_logger().error(f"Config file not found: {config_path}")
                raise FileNotFoundError(f"Config file not found: {config_path}")
            
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            self.get_logger().info(f"Loaded config from: {config_path}")
            return config
            
        except Exception as e:
            self.get_logger().error(f"Failed to load config: {str(e)}")
            raise
    
    def load_model(self):
        """Load Model from configured path"""
        try:
            model_path = os.path.expanduser(self.config['model']['model_dir'])
            
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file not found: {model_path}")
            
            # Parse device
            device = self.parse_device(self.config['model']['device'])
            
            self.model = YOLO(model_path)
            self.get_logger().info(f"Loaded model from: {model_path}")
            self.get_logger().info(f"Using device: {device}")
            
            # Set device
            self.model.to(device)
            
        except Exception as e:
            self.get_logger().error(f"Failed to load model: {str(e)}")
            raise
    
    def log_config(self):
        """Log important configuration parameters"""
        self.get_logger().info('=== Configuration ===')
        self.get_logger().info(f'Model: {self.config["model"]["model_dir"]}')
        self.get_logger().info(f'Dataset: {self.config["dataset"]["path"]}')
        self.get_logger().info(f'Device: {self.config["model"]["device"]}')
        self.get_logger().info(f'Confidence: {self.config["inference"].get("conf", 0.25)}')
        self.get_logger().info(f'IoU: {self.config["inference"].get("iou", 0.45)}')
    
    def parse_device(self, device):
        """Parse device configuration"""
        if device == '-1':
            return -1  # Most idle GPU
        elif isinstance(device, str) and ',' in device:
            return [int(d.strip()) for d in device.split(',')]
        elif device == 'cpu':
            return 'cpu'
        else:
            try:
                return int(device)
            except:
                return device
    
    #################### Image processing ####################

    def run_inference(self, image):
        """Run inference on a single image"""
        # Verify image is good
        if image is None or image.size == 0:
            print("Invalid image for inference, skipping.")
            return
        
        # Run inference
        result = self.model.predict(
            image,
            conf=self.conf,  # Minimum confidence
            iou=self.iou,  # Minimum IoU for NMS
            imgsz=self.imgsz,  # Inference image size
            max_det=self.max_det,  # Maximum detections per image
            verbose=False 
        )
        
        # Get the encoded mask
        mask_with_confidence = self.merge_confidence_and_masks(result[0])
        
        # Publish the mask
        self.publish_mask_with_confidence(mask_with_confidence)

    def merge_confidence_and_masks(self, result):
        """
        Merge segmentation masks with confidence scores into a single 16UC1 image.
        
        Encoding format (16-bit):
        - Upper 8 bits: class_id (0-255)
        - Lower 8 bits: confidence (0-255, representing 0.0-1.0)
        - Background pixels (no detection): 0
        
        For overlapping regions, the class with higher confidence is kept.
        
        Example: class_id=5, confidence=0.85
        -> encoded = (5 << 8) | int(0.85 * 255) = 1280 | 216 = 1496
        
        Returns:
            np.ndarray: 16-bit unsigned integer image (height, width) with encoded class and confidence
        """
        # Get image dimensions from the original image
        orig_shape = result.orig_shape  # (height, width)
        height, width = orig_shape
        
        # Initialize output mask as 16-bit unsigned integer
        mask_with_confidence = np.zeros((height, width), dtype=np.uint16)
        
        # Create a separate confidence map to track the highest confidence at each pixel
        confidence_map = np.zeros((height, width), dtype=np.float32)
        
        # Check if any masks were detected
        if result.masks is None or len(result.masks) == 0:
            self.get_logger().warn("No masks detected in image")
            return mask_with_confidence
        
        # Get masks and corresponding data
        masks = result.masks.data  # Tensor of shape (N, H, W) where N is number of detections
        boxes = result.boxes  # Boxes object containing cls and conf
        
        # Convert masks to numpy
        masks_np = masks.cpu().numpy()
        
        # Iterate through each detection
        for i in range(len(masks_np)):
            # Get class and confidence for this detection
            class_id = int(boxes.cls[i].item())
            confidence = float(boxes.conf[i].item())
            
            # Get the mask for this detection (binary mask)
            mask = masks_np[i]
            
            # Resize mask to original image size if necessary
            if mask.shape != (height, width):
                mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
            
            # Convert to binary mask (threshold at 0.5)
            binary_mask = (mask > 0.5).astype(bool)
            
            # Update pixels where this mask has higher confidence
            update_mask = binary_mask & (confidence > confidence_map)
            
            # Encode class and confidence into uint16 using bit-shift
            # Upper 8 bits: class_id, Lower 8 bits: confidence
            encoded_value = (class_id << 8) | int(confidence * 255)
            
            # Apply to output mask only where we should update
            mask_with_confidence[update_mask] = encoded_value
            
            # Update confidence map for future comparisons
            confidence_map[update_mask] = confidence
                
        return mask_with_confidence
    
def main(args=None):
    rclpy.init(args=args)
    node = TAMTSegmentationNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down TAMT Segmentation Node...")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()