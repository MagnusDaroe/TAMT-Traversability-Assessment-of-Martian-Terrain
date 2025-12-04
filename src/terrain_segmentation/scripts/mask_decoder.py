#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np


class SegmentationViewerNode(Node):
    def __init__(self):
        super().__init__('segmentation_viewer_node')
        
        # Initialize CV bridge
        self.bridge = CvBridge()
        
        # # Class names 
        # self.class_names = {
        #     0: "background",
        #     1: "soil",
        #     2: "bedrock",
        #     3: "sand",
        #     4: "big_rock"
        # }

          # Class names 
        self.class_names = {
            0: "soil",
            1: "bedrock",
            2: "sand",
            3: "big_rock"
        }


        # names:
        # 0: soil
        # 1: bedrock
        # 2: sand
        # 3: big_rock
        
        # Color palette for visualization
        np.random.seed(42)
        self.colors = np.random.randint(0, 255, size=(256, 3), dtype=np.uint8)
        
        # ------ Subscribers ------
        self.mask_sub = self.create_subscription(
            Image,
            'tamt/segmentation/masks_with_confidence',
            self.mask_callback,
            10
        )
        
        # Also subscribe to original image if you want side-by-side view
        self.image_sub = self.create_subscription(
            Image,
            'tamt/camera/image_raw',
            self.image_callback,
            10
        )
        
        # Store latest images and visualizations
        self.latest_mask = None
        self.latest_image = None
        self.current_visualizations = {}
        
        # Display mode: 'overlay', 'class', 'confidence', 'all'
        self.display_mode = 'overlay'
        
        # Create OpenCV windows
        cv2.namedWindow("Segmentation Overlay", cv2.WINDOW_NORMAL)
        cv2.namedWindow("Class Segmentation", cv2.WINDOW_NORMAL)
        cv2.namedWindow("Confidence Map", cv2.WINDOW_NORMAL)
        
        # Timer for handling OpenCV window updates and keyboard input
        self.timer = self.create_timer(0.033, self.timer_callback)  # ~30 FPS
        
        self.get_logger().info('Segmentation Viewer Node initialized')
        self.get_logger().info('Subscribing to: tamt/segmentation/masks_with_confidence')
        self.get_logger().info('Press "q" to quit, "c" for class view, "f" for confidence view, "o" for overlay, "a" for all')

    def image_callback(self, msg):
        """Store the latest original image"""
        try:
            self.latest_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f"Failed to convert image: {str(e)}")

    def mask_callback(self, msg):
        """Receive and process the segmentation mask"""
        try:
            # Convert ROS Image message to OpenCV format (16UC1)
            encoded_mask = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
            
            if encoded_mask is None or encoded_mask.size == 0:
                self.get_logger().warn("Received empty mask")
                return
            
            # Store latest mask
            self.latest_mask = encoded_mask
            
            # Decode the mask
            class_mask, confidence_mask = self.decode_mask(encoded_mask)
            
            # Create visualizations (don't display yet, just prepare them)
            self.prepare_visualizations(class_mask, confidence_mask)
            
        except Exception as e:
            self.get_logger().error(f"Failed to process mask: {str(e)}")

    def timer_callback(self):
        """Handle OpenCV window updates and keyboard input"""
        # Display the prepared visualizations
        self.display_visualizations()
        
        # Handle keyboard input (1ms wait to not block)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            self.get_logger().info("Quit requested")
            rclpy.shutdown()
        elif key == ord('c'):
            self.display_mode = 'class'
            self.get_logger().info("Display mode: Class")
        elif key == ord('f'):
            self.display_mode = 'confidence'
            self.get_logger().info("Display mode: Confidence")
        elif key == ord('o'):
            self.display_mode = 'overlay'
            self.get_logger().info("Display mode: Overlay")
        elif key == ord('a'):
            self.display_mode = 'all'
            self.get_logger().info("Display mode: All")

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

    def prepare_visualizations(self, class_mask, confidence_mask):
        """Create visualizations and store them for display"""
        
        # Get unique classes (excluding background)
        unique_classes = np.unique(class_mask)
        unique_classes = unique_classes[unique_classes != 0]
        
        height, width = class_mask.shape
        
        # Clear previous visualizations
        self.current_visualizations = {}
        
        # Create class visualization
        if self.display_mode == 'class' or self.display_mode == 'all':
            self.current_visualizations['class'] = self.create_class_visualization(class_mask, unique_classes)
        
        # Create confidence visualization
        if self.display_mode == 'confidence' or self.display_mode == 'all':
            self.current_visualizations['confidence'] = self.create_confidence_visualization(confidence_mask)
        
        # Create overlay visualization
        if self.display_mode == 'overlay' or self.display_mode == 'all':
            if self.latest_image is not None:
                self.current_visualizations['overlay'] = self.create_overlay_visualization(
                    self.latest_image, class_mask, confidence_mask, unique_classes
                )
            else:
                # Create colored visualization without original image
                self.current_visualizations['overlay'] = self.create_class_visualization(class_mask, unique_classes)

    def display_visualizations(self):
        """Display all prepared visualizations"""
        if 'class' in self.current_visualizations:
            cv2.imshow("Class Segmentation", self.current_visualizations['class'])
        
        if 'confidence' in self.current_visualizations:
            cv2.imshow("Confidence Map", self.current_visualizations['confidence'])
        
        if 'overlay' in self.current_visualizations:
            cv2.imshow("Segmentation Overlay", self.current_visualizations['overlay'])

    def create_class_visualization(self, class_mask, unique_classes):
        """Create colored visualization of classes"""
        height, width = class_mask.shape
        vis_image = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Color each class
        for class_id in unique_classes:
            color = self.colors[int(class_id) % 256].tolist()
            vis_image[class_mask == class_id] = color
        
        # Add legend
        vis_image = self.add_legend(vis_image, unique_classes)
        
        return vis_image

    def create_confidence_visualization(self, confidence_mask):
        """Create heatmap visualization of confidence"""
        # Convert confidence to color (0.0 = blue/cold, 1.0 = red/hot)
        conf_vis = (confidence_mask * 255).astype(np.uint8)
        conf_vis = cv2.applyColorMap(conf_vis, cv2.COLORMAP_JET)
        
        # Add color bar legend
        conf_vis = self.add_colorbar(conf_vis)
        
        return conf_vis

    def create_overlay_visualization(self, original_image, class_mask, confidence_mask, unique_classes):
        """Create overlay of segmentation on original image"""
        # Resize original image if needed
        if original_image.shape[:2] != class_mask.shape:
            original_image = cv2.resize(original_image, (class_mask.shape[1], class_mask.shape[0]))
        
        overlay = original_image.copy()
        
        # Process each class
        for class_id in unique_classes:
            # Get mask for this class
            binary_mask = (class_mask == class_id).astype(np.uint8)
            
            # Get average confidence for this class
            class_confidences = confidence_mask[binary_mask == 1]
            avg_confidence = np.mean(class_confidences) if len(class_confidences) > 0 else 0.0
            
            # Get color for this class
            color = self.colors[int(class_id) % 256].tolist()
            
            # Create colored mask
            height, width = class_mask.shape
            colored_mask = np.zeros((height, width, 3), dtype=np.uint8)
            colored_mask[binary_mask == 1] = color
            
            # Blend with original image
            overlay = cv2.addWeighted(overlay, 1.0, colored_mask, 0.5, 0)
            
            # Add text label with class and confidence
            y_indices, x_indices = np.where(binary_mask == 1)
            if len(y_indices) > 0:
                label_y = int(np.min(y_indices))
                label_x = int(np.min(x_indices))
                
                # Get class name
                class_name = self.class_names.get(class_id, f"class_{class_id}")
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

    def add_legend(self, image, unique_classes):
        """Add legend showing class colors and names"""
        # Match legend height to image height
        legend_height = image.shape[0]
        legend_width = 200
        
        # Create legend area
        legend = np.zeros((legend_height, legend_width, 3), dtype=np.uint8)
        
        # Add title
        cv2.putText(legend, "Classes:", (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Add each class
        for idx, class_id in enumerate(unique_classes):
            y_pos = 40 + idx * 30
            color = self.colors[int(class_id) % 256].tolist()
            class_name = self.class_names.get(class_id, f"class_{class_id}")
            
            # Draw color box
            cv2.rectangle(legend, (10, y_pos - 10), (30, y_pos + 5), color, -1)
            
            # Draw text
            cv2.putText(legend, class_name, (40, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Attach legend to the right of the image
        combined = np.hstack([image, legend])
        return combined
    
    def add_colorbar(self, image):
        """Add colorbar for confidence visualization"""
        colorbar_width = 50
        colorbar_height = image.shape[0]
        
        # Create colorbar
        colorbar = np.linspace(255, 0, colorbar_height, dtype=np.uint8)
        colorbar = np.tile(colorbar.reshape(-1, 1), (1, colorbar_width))
        colorbar = cv2.applyColorMap(colorbar, cv2.COLORMAP_JET)
        
        # Add labels
        cv2.putText(colorbar, "1.0", (5, 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.putText(colorbar, "0.5", (5, colorbar_height // 2),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.putText(colorbar, "0.0", (5, colorbar_height - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Attach colorbar to the right of the image
        combined = np.hstack([image, colorbar])
        return combined


def main(args=None):
    rclpy.init(args=args)
    node = SegmentationViewerNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down Segmentation Viewer Node...")
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()