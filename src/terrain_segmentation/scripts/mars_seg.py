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


class YOLOInferenceNode(Node):
    def __init__(self):
        super().__init__('yolo_inference_node')
        
        # Declare parameter for config file
        self.declare_parameter('config_file', 'seg_inference.yaml')
        config_file = self.get_parameter('config_file').value
        
        # Load configuration
        self.config = self.load_config(config_file)
        
        # Initialize CV bridge
        self.bridge = CvBridge()
        
        # Load model
        self.model = None
        self.load_model()
        
        # Load dataset images
        self.image_paths = []
        self.current_index = 0
        self.current_result = None
        
        self.load_dataset_images()
        
        # Publishers
        self.status_pub = self.create_publisher(String, 'yolo_inference/status', 10)
        self.image_pub = self.create_publisher(Image, 'yolo_inference/image', 10)
        self.result_pub = self.create_publisher(Image, 'yolo_inference/result', 10)
        
        # GUI
        self.gui_thread = None
        self.root = None
        
        self.get_logger().info('YOLO Inference Node initialized')
        self.log_config()
        
        # Start GUI
        self.start_gui()
        
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
        """Load YOLO model from configured path"""
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
    
    def load_dataset_images(self):
        """Load all images from the dataset directory"""
        try:
            dataset_path = os.path.expanduser(self.config['dataset']['path'])
            
            if not os.path.exists(dataset_path):
                raise FileNotFoundError(f"Dataset path not found: {dataset_path}")
            
            # Supported image extensions
            image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp')
            
            # Recursively find all images
            self.image_paths = []
            for root, dirs, files in os.walk(dataset_path):
                for file in sorted(files):
                    if file.lower().endswith(image_extensions):
                        full_path = os.path.join(root, file)
                        self.image_paths.append(full_path)
            
            if not self.image_paths:
                raise ValueError(f"No images found in dataset path: {dataset_path}")
            
            self.get_logger().info(f"Found {len(self.image_paths)} images in dataset")
            self.get_logger().info(f"First image: {self.image_paths[0]}")
            
        except Exception as e:
            self.get_logger().error(f"Failed to load dataset images: {str(e)}")
            raise
    
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
    
    def run_inference(self, image_path):
        """Run inference on a single image"""
        try:
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Failed to load image: {image_path}")
            
            # Get inference parameters
            conf = self.config['inference'].get('conf', 0.25)
            iou = self.config['inference'].get('iou', 0.45)
            imgsz = self.config['inference'].get('imgsz', 640)
            max_det = self.config['inference'].get('max_det', 300)
            
            # Run inference
            results = self.model.predict(
                image,
                conf=conf,
                iou=iou,
                imgsz=imgsz,
                max_det=max_det,
                verbose=False
            )
            
            self.current_result = results[0]
            
            # Get annotated image
            annotated = results[0].plot()
            
            return image, annotated, results[0]
            
        except Exception as e:
            self.get_logger().error(f"Inference failed: {str(e)}")
            raise
    
    def log_config(self):
        """Log important configuration parameters"""
        self.get_logger().info('=== Configuration ===')
        self.get_logger().info(f'Model: {self.config["model"]["model_dir"]}')
        self.get_logger().info(f'Dataset: {self.config["dataset"]["path"]}')
        self.get_logger().info(f'Device: {self.config["model"]["device"]}')
        self.get_logger().info(f'Confidence: {self.config["inference"].get("conf", 0.25)}')
        self.get_logger().info(f'IoU: {self.config["inference"].get("iou", 0.45)}')
    
    def start_gui(self):
        """Start the GUI in a separate thread"""
        self.gui_thread = threading.Thread(target=self.run_gui, daemon=True)
        self.gui_thread.start()
    
    def run_gui(self):
        """Run the GUI main loop"""
        self.root = tk.Tk()
        self.root.title("YOLO Inference Viewer")
        self.root.geometry("1400x800")
        
        # Create main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(1, weight=1)
        
        # Title
        title_label = ttk.Label(main_frame, text="YOLO Inference Test", font=('Arial', 16, 'bold'))
        title_label.grid(row=0, column=0, columnspan=2, pady=10)
        
        # Image frames
        original_frame = ttk.LabelFrame(main_frame, text="Original Image", padding="5")
        original_frame.grid(row=1, column=0, padx=5, pady=5, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        result_frame = ttk.LabelFrame(main_frame, text="Inference Result", padding="5")
        result_frame.grid(row=1, column=1, padx=5, pady=5, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Image labels
        self.original_label = ttk.Label(original_frame)
        self.original_label.pack(expand=True, fill=tk.BOTH)
        
        self.result_label = ttk.Label(result_frame)
        self.result_label.pack(expand=True, fill=tk.BOTH)
        
        # Info frame
        info_frame = ttk.Frame(main_frame)
        info_frame.grid(row=2, column=0, columnspan=2, pady=10)
        
        self.info_label = ttk.Label(info_frame, text="", font=('Arial', 10))
        self.info_label.pack()
        
        self.detection_label = ttk.Label(info_frame, text="", font=('Arial', 9))
        self.detection_label.pack()
        
        # Control frame
        control_frame = ttk.Frame(main_frame)
        control_frame.grid(row=3, column=0, columnspan=2, pady=10)
        
        # Navigation buttons
        self.prev_button = ttk.Button(control_frame, text="◄ Previous", command=self.prev_image, width=15)
        self.prev_button.pack(side=tk.LEFT, padx=5)
        
        self.index_label = ttk.Label(control_frame, text="", font=('Arial', 10))
        self.index_label.pack(side=tk.LEFT, padx=20)
        
        self.next_button = ttk.Button(control_frame, text="Next ►", command=self.next_image, width=15)
        self.next_button.pack(side=tk.LEFT, padx=5)
        
        # Separator
        ttk.Separator(control_frame, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=20)
        
        # Run inference button
        self.infer_button = ttk.Button(control_frame, text="Run Inference", command=self.run_current_inference, width=15)
        self.infer_button.pack(side=tk.LEFT, padx=5)
        
        # Save result button
        self.save_button = ttk.Button(control_frame, text="Save Result", command=self.save_result, width=15)
        self.save_button.pack(side=tk.LEFT, padx=5)
        
        # Keyboard bindings
        self.root.bind('<Left>', lambda e: self.prev_image())
        self.root.bind('<Right>', lambda e: self.next_image())
        self.root.bind('<space>', lambda e: self.run_current_inference())
        self.root.bind('s', lambda e: self.save_result())
        
        # Load first image
        self.update_display()
        
        # Start main loop
        self.root.mainloop()
    
    def prev_image(self):
        """Navigate to previous image"""
        if self.current_index > 0:
            self.current_index -= 1
            self.update_display()
    
    def next_image(self):
        """Navigate to next image"""
        if self.current_index < len(self.image_paths) - 1:
            self.current_index += 1
            self.update_display()
    
    def update_display(self):
        """Update the display with current image"""
        if not self.image_paths:
            return
        
        try:
            # Load and display original image
            image_path = self.image_paths[self.current_index]
            image = cv2.imread(image_path)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Resize for display
            display_size = (640, 480)
            image_resized = cv2.resize(image_rgb, display_size)
            
            # Convert to PIL and then to PhotoImage
            pil_image = PILImage.fromarray(image_resized)
            photo = ImageTk.PhotoImage(pil_image)
            
            self.original_label.configure(image=photo)
            self.original_label.image = photo  # Keep a reference
            
            # Clear result
            self.result_label.configure(image='')
            self.result_label.image = None
            
            # Update info
            rel_path = os.path.relpath(image_path, os.path.expanduser(self.config['dataset']['path']))
            self.info_label.configure(text=f"File: {rel_path}")
            self.index_label.configure(text=f"{self.current_index + 1} / {len(self.image_paths)}")
            self.detection_label.configure(text="Press 'Run Inference' or Space to analyze")
            
            # Update button states
            self.prev_button.configure(state=tk.NORMAL if self.current_index > 0 else tk.DISABLED)
            self.next_button.configure(state=tk.NORMAL if self.current_index < len(self.image_paths) - 1 else tk.DISABLED)
            
        except Exception as e:
            self.get_logger().error(f"Failed to update display: {str(e)}")
    
    def run_current_inference(self):
        """Run inference on current image"""
        if not self.image_paths:
            return
        
        try:
            self.infer_button.configure(state=tk.DISABLED)
            self.info_label.configure(text="Running inference...")
            self.root.update()
            
            image_path = self.image_paths[self.current_index]
            original, annotated, result = self.run_inference(image_path)
            
            # Convert annotated image for display
            annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            display_size = (640, 480)
            annotated_resized = cv2.resize(annotated_rgb, display_size)
            
            pil_result = PILImage.fromarray(annotated_resized)
            photo_result = ImageTk.PhotoImage(pil_result)
            
            self.result_label.configure(image=photo_result)
            self.result_label.image = photo_result
            
            # Update detection info
            if hasattr(result, 'boxes') and result.boxes is not None:
                num_detections = len(result.boxes)
                
                # Get class distribution
                if hasattr(result.boxes, 'cls'):
                    classes = result.boxes.cls.cpu().numpy()
                    class_counts = {}
                    for cls_id in classes:
                        cls_name = result.names[int(cls_id)]
                        class_counts[cls_name] = class_counts.get(cls_name, 0) + 1
                    
                    detection_text = f"Detections: {num_detections} | "
                    detection_text += " | ".join([f"{name}: {count}" for name, count in class_counts.items()])
                else:
                    detection_text = f"Detections: {num_detections}"
            elif hasattr(result, 'masks') and result.masks is not None:
                num_masks = len(result.masks)
                detection_text = f"Segmentation masks: {num_masks}"
            else:
                detection_text = "No detections"
            
            self.detection_label.configure(text=detection_text)
            
            rel_path = os.path.relpath(image_path, os.path.expanduser(self.config['dataset']['path']))
            self.info_label.configure(text=f"File: {rel_path}")
            
            # Publish results
            self.publish_results(original, annotated)
            
            # Publish status
            status_msg = String()
            status_msg.data = f"Inference complete: {detection_text}"
            self.status_pub.publish(status_msg)
            
            self.get_logger().info(f"Inference complete: {detection_text}")
            
        except Exception as e:
            self.get_logger().error(f"Inference failed: {str(e)}")
            self.detection_label.configure(text=f"Error: {str(e)}")
        
        finally:
            self.infer_button.configure(state=tk.NORMAL)
    
    def save_result(self):
        """Save the current result image"""
        if self.current_result is None:
            self.detection_label.configure(text="No result to save. Run inference first.")
            return
        
        try:
            # Get save directory from config
            save_dir = os.path.expanduser(self.config['inference'].get('save_dir', '~/inference_results'))
            os.makedirs(save_dir, exist_ok=True)
            
            # Get current image name
            image_path = self.image_paths[self.current_index]
            image_name = os.path.basename(image_path)
            name_without_ext = os.path.splitext(image_name)[0]
            
            # Save result
            save_path = os.path.join(save_dir, f"{name_without_ext}_result.jpg")
            
            # Get the plotted result
            annotated = self.current_result.plot()
            cv2.imwrite(save_path, annotated)
            
            self.detection_label.configure(text=f"Saved to: {save_path}")
            self.get_logger().info(f"Result saved to: {save_path}")
            
        except Exception as e:
            self.get_logger().error(f"Failed to save result: {str(e)}")
            self.detection_label.configure(text=f"Error saving: {str(e)}")
    
    def publish_results(self, original, annotated):
        """Publish original and result images to ROS topics"""
        try:
            # Publish original image
            original_msg = self.bridge.cv2_to_imgmsg(original, encoding="bgr8")
            self.image_pub.publish(original_msg)
            
            # Publish result image
            result_msg = self.bridge.cv2_to_imgmsg(annotated, encoding="bgr8")
            self.result_pub.publish(result_msg)
            
        except Exception as e:
            self.get_logger().error(f"Failed to publish images: {str(e)}")
    
    def cleanup(self):
        """Cleanup resources"""
        if self.root:
            self.root.quit()


def main(args=None):
    rclpy.init(args=args)
    node = YOLOInferenceNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down YOLO Inference Node...")
    finally:
        node.cleanup()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()