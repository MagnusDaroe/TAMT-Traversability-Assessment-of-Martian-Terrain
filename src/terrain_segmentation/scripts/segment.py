#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from ultralytics import YOLO
import os
import yaml
import cv2
import numpy as np
from pathlib import Path
from ament_index_python.packages import get_package_share_directory
import tkinter as tk
from tkinter import ttk
from PIL import Image as PILImage, ImageTk
import threading


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
                
        # ------ Publishers ------
        self.segmentation_mask_pub = self.create_publisher(Image, 'tamt/segmentation/masks_with_confidence', 10)
      
        # ------ Subscribers ------
        self.image_sub = self.create_subscription(
            Image,
            'tamt/camera/image_raw',
            self.image_callback,
            10
        )
 
        self.get_logger().info('TAMT Segmentation Node initialized')
        self.log_config()
    
    #################### Callbacks ####################

    def image_callback(self, msg):
        pass  # To be implemented: process incoming images and publish segmentation masks

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
        self.publish_results(mask_with_confidence)


        # Decode back to class and confidence masks
        class_mask, confidence_mask = self.decode_mask(mask_with_confidence)
        
        # ===== Create visualizations =====
        # 1. YOLO's annotated output
        yolo_annotated = result[0].plot()
        
        # 2. Our encoded/decoded visualization (matching YOLO's style)
        decoded_vis = self.visualize_decoded_masks(image, class_mask, confidence_mask)
        
        # Combine side by side
        comparison = self.create_side_by_side_comparison(yolo_annotated, decoded_vis)
        
        # Display
        cv2.imshow("YOLO vs Encoded/Decoded Comparison", comparison)
        cv2.waitKey(0)


    def visualize_decoded_masks(self, original_image, class_mask, confidence_mask):
        """
        Visualize decoded masks in YOLO's style (overlay on original image with labels)
        """
        overlay = original_image.copy()
        height, width = class_mask.shape
        
        # Get unique classes (excluding background)
        unique_classes = np.unique(class_mask)
        unique_classes = unique_classes[unique_classes != 0]
        
        if len(unique_classes) == 0:
            cv2.putText(overlay, "No masks detected", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            return overlay
        
        # Define colors for different classes (same seed as YOLO uses)
        np.random.seed(42)
        colors = np.random.randint(0, 255, size=(256, 3), dtype=np.uint8)
        
        # Process each class
        for class_id in unique_classes:
            # Get mask for this class
            binary_mask = (class_mask == class_id).astype(np.uint8)
            
            # Get average confidence for this class
            class_confidences = confidence_mask[binary_mask == 1]
            avg_confidence = np.mean(class_confidences) if len(class_confidences) > 0 else 0.0
            
            # Get color for this class
            color_idx = int(class_id) % 256
            color = colors[color_idx].tolist()
            
            # Create colored mask
            colored_mask = np.zeros((height, width, 3), dtype=np.uint8)
            colored_mask[binary_mask == 1] = color
            
            # Blend with original image
            overlay = cv2.addWeighted(overlay, 1.0, colored_mask, 0.5, 0)
            
            # Add text label with class and confidence
            # Find top-left corner of mask for label placement
            y_indices, x_indices = np.where(binary_mask == 1)
            if len(y_indices) > 0:
                label_y = int(np.min(y_indices))
                label_x = int(np.min(x_indices))
                
                # Get class name if available, otherwise use ID
                class_name = self.get_class_name(class_id)
                label = f"{class_name} {avg_confidence:.2f}"
                
                # Add background rectangle for text
                (text_width, text_height), _ = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2
                )
                cv2.rectangle(
                    overlay,
                    (label_x, label_y - text_height - 5),
                    (label_x + text_width, label_y),
                    color,
                    -1
                )
                
                # Add text
                cv2.putText(
                    overlay, label, (label_x, label_y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2
                )
        
        return overlay


    def get_class_name(self, class_id):
        """Get class name from model or return generic name"""
        try:
            if hasattr(self.model, 'names') and class_id in self.model.names:
                return self.model.names[class_id]
        except:
            pass
        return f"class_{class_id}"


    def create_side_by_side_comparison(self, yolo_image, decoded_image):
        """
        Create side-by-side comparison of YOLO output and decoded output
        """
        # Ensure both images are the same height
        h1, w1 = yolo_image.shape[:2]
        h2, w2 = decoded_image.shape[:2]
        
        # Use the maximum height
        max_height = max(h1, h2)
        
        # Resize if needed
        if h1 != max_height:
            aspect_ratio = w1 / h1
            yolo_image = cv2.resize(yolo_image, (int(max_height * aspect_ratio), max_height))
        
        if h2 != max_height:
            aspect_ratio = w2 / h2
            decoded_image = cv2.resize(decoded_image, (int(max_height * aspect_ratio), max_height))
        
        # Add labels on top of each image
        yolo_labeled = yolo_image.copy()
        decoded_labeled = decoded_image.copy()
        
        # Add title backgrounds
        cv2.rectangle(yolo_labeled, (0, 0), (yolo_labeled.shape[1], 40), (0, 0, 0), -1)
        cv2.rectangle(decoded_labeled, (0, 0), (decoded_labeled.shape[1], 40), (0, 0, 0), -1)
        
        # Add titles
        cv2.putText(yolo_labeled, "YOLO Original", (10, 28),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(decoded_labeled, "Encoded/Decoded", (10, 28),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        
        # Combine side by side
        combined = np.hstack([yolo_labeled, decoded_labeled])
        
        return combined


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
            
            # Only update pixels where:
            # 1. The binary mask is True (this detection covers this pixel)
            # 2. The new confidence is higher than existing confidence at that pixel
            update_mask = binary_mask & (confidence > confidence_map)
            
            # Encode class and confidence into uint16 using bit-shift
            # Upper 8 bits: class_id, Lower 8 bits: confidence
            encoded_value = (class_id << 8) | int(confidence * 255)
            
            # Apply to output mask only where we should update
            mask_with_confidence[update_mask] = encoded_value
            
            # Update confidence map for future comparisons
            confidence_map[update_mask] = confidence
        
        self.get_logger().info(
            f"Created mask with {len(masks_np)} detections, "
            f"shape: {mask_with_confidence.shape}, "
            f"dtype: {mask_with_confidence.dtype}, "
            f"non-zero pixels: {np.count_nonzero(mask_with_confidence)}"
        )
        
        return mask_with_confidence
    
    def decode_mask(self, encoded_mask):
        """
        Decode the 16UC1 mask back into class IDs and confidence values.
        
        Args:
            encoded_mask: uint16 numpy array with encoded values

        Returns:
            class_mask: Class ID for each pixel (0 = background) as uint8
            confidence_mask: Confidence value for each pixel (0.0-1.0) as float32
        """
        # Extract upper 8 bits for class_id
        class_mask = (encoded_mask >> 8).astype(np.uint8)

        # Extract lower 8 bits for confidence and normalize to 0.0-1.0
        confidence_mask = ((encoded_mask & 0xFF) / 255.0).astype(np.float32)

        return class_mask, confidence_mask
    

    def publish_results(self, mask_with_confidence):
        """Publish segmentation mask to ROS topic"""
        try:
            # Convert to ROS Image message with 16UC1 encoding
            mask_msg = self.bridge.cv2_to_imgmsg(mask_with_confidence, encoding="16UC1")
            self.segmentation_mask_pub.publish(mask_msg)

            # Display the raw image
            currentimage = mask_with_confidence.copy()
            cv2.imshow("Raw Image", currentimage)
            cv2.waitKey(10)

            self.get_logger().info("Published segmentation mask")
            
        except Exception as e:
            self.get_logger().error(f"Failed to publish mask: {str(e)}")
    

def main(args=None):
    rclpy.init(args=args)
    node = TAMTSegmentationNode()
    
    # Load image
    image_path = '/home/daroe/tamt/dataset/AI4MARS_NAV_GOOD/images/m2020_test_NLF_0013_0668101876_305ECM_N0030028NCAM00104_01_295J_merged63.jpeg'
    image = cv2.imread(image_path)

    node.run_inference(image)

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down TAMT Segmentation Node...")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()