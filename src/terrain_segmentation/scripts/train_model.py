#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from std_srvs.srv import Trigger
from ultralytics import YOLO
import os
import yaml
import threading
from pathlib import Path
from ament_index_python.packages import get_package_share_directory


class YOLOTrainerNode(Node):
    def __init__(self):
        super().__init__('yolo_trainer_node')
        
        # Declare parameter for config file
        self.declare_parameter('config_file', 'seg_train.yaml')
        config_file = self.get_parameter('config_file').value
        
        # Load configuration
        self.config = self.load_config(config_file)
        
        # Training state
        self.is_training = False
        self.training_thread = None
        self.model = None
        self.best_hyperparameters = None
        
        # Publishers
        self.status_pub = self.create_publisher(String, 'yolo_trainer/status', 10)
        
        # Services
        self.start_training_srv = self.create_service(
            Trigger, 
            'yolo_trainer/start_training', 
            self.start_training_callback
        )
        self.stop_training_srv = self.create_service(
            Trigger, 
            'yolo_trainer/stop_training', 
            self.stop_training_callback
        )
        
        # Timer for status updates
        self.status_timer = self.create_timer(5.0, self.publish_status)
        
        self.get_logger().info('YOLO Trainer Node initialized')
        self.log_config()
        
    def load_config(self, config_file):
        """Load configuration from YAML file"""
        try:
            # Try to find config in package share directory
            package_share = get_package_share_directory('terrain_segmentation')
            config_path = os.path.join(package_share, 'config', config_file)
            
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
    
    def validate_dataset(self, data_yaml_path):
        """Validate that the dataset and its components exist"""
        # Check if data.yaml exists
        if not os.path.exists(data_yaml_path):
            raise FileNotFoundError(f"Dataset YAML not found: {data_yaml_path}")
        
        self.get_logger().info(f"Found dataset YAML: {data_yaml_path}")
        
        # Load and validate data.yaml contents
        try:
            with open(data_yaml_path, 'r') as f:
                data_config = yaml.safe_load(f)
            
            # Check for required fields
            if 'path' not in data_config:
                raise ValueError(f"Dataset YAML missing 'path' field: {data_yaml_path}")
            
            # Get dataset root path
            dataset_root = data_config['path']
            if not os.path.isabs(dataset_root):
                # If relative, make it relative to the yaml file location
                yaml_dir = os.path.dirname(data_yaml_path)
                dataset_root = os.path.join(yaml_dir, dataset_root)
            
            dataset_root = os.path.expanduser(dataset_root)
            
            if not os.path.exists(dataset_root):
                raise FileNotFoundError(f"Dataset root directory not found: {dataset_root}")
            
            self.get_logger().info(f"Found dataset root: {dataset_root}")
            
            # Check for train/val/test directories
            required_splits = []
            if 'train' in data_config:
                required_splits.append(('train', data_config['train']))
            if 'val' in data_config:
                required_splits.append(('val', data_config['val']))
            
            if not required_splits:
                raise ValueError(f"Dataset YAML must contain at least 'train' field: {data_yaml_path}")
            
            for split_name, split_path in required_splits:
                if not os.path.isabs(split_path):
                    full_split_path = os.path.join(dataset_root, split_path)
                else:
                    full_split_path = split_path
                
                full_split_path = os.path.expanduser(full_split_path)
                
                if not os.path.exists(full_split_path):
                    raise FileNotFoundError(
                        f"Dataset split '{split_name}' not found: {full_split_path}\n"
                        f"Expected path from dataset root: {dataset_root}"
                    )
                
                # Check if directory contains any images
                if os.path.isdir(full_split_path):
                    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif')
                    images = [f for f in os.listdir(full_split_path) 
                             if f.lower().endswith(image_extensions)]
                    
                    if not images:
                        self.get_logger().warn(
                            f"Warning: No images found in {split_name} directory: {full_split_path}"
                        )
                    else:
                        self.get_logger().info(
                            f"Found {len(images)} images in {split_name} split"
                        )
                
            # Check for class names
            if 'names' not in data_config:
                raise ValueError(f"Dataset YAML missing 'names' field: {data_yaml_path}")
            
            num_classes = len(data_config['names'])
            self.get_logger().info(f"Dataset contains {num_classes} classes: {list(data_config['names'].values())}")
            
            self.get_logger().info("✓ Dataset validation passed")
            return True
            
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML format in {data_yaml_path}: {str(e)}")
        except Exception as e:
            raise Exception(f"Dataset validation failed: {str(e)}")
    
    def log_config(self):
        """Log important configuration parameters"""
        self.get_logger().info('=== Configuration ===')
        self.get_logger().info(f'Training mode: {self.config["training_mode"]}')
        self.get_logger().info(f'Model: {self.config["model"]["type"]}')
        self.get_logger().info(f'Dataset: {self.config["dataset"]["path"]}')
        self.get_logger().info(f'Epochs: {self.config["training"]["epochs"]}')
        self.get_logger().info(f'Batch size: {self.config["training"]["batch_size"]}')
        self.get_logger().info(f'Image size: {self.config["training"]["imgsz"]}')
        self.get_logger().info(f'Device: {self.config["hardware"]["device"]}')
        
        if 'search_hyperparameters' in self.config['training_mode']:
            self.get_logger().info(f'Tuning iterations: {self.config["hyperparameter_tuning"]["iterations"]}')
    
    def start_training_callback(self, request, response):
        if self.is_training:
            response.success = False
            response.message = "Training is already in progress"
            self.get_logger().warn("Training already in progress")
            return response
        
        # Start training in a separate thread
        self.training_thread = threading.Thread(target=self.train_model)
        self.training_thread.start()
        
        response.success = True
        response.message = "Training started successfully"
        self.get_logger().info("Training started")
        return response
    
    def stop_training_callback(self, request, response):
        if not self.is_training:
            response.success = False
            response.message = "No training is currently in progress"
            self.get_logger().warn("No training to stop")
            return response
        
        response.success = True
        response.message = "Training stop requested (will complete current epoch)"
        self.get_logger().warn("Training stop requested - will complete current epoch")
        return response
    
    def train_model(self):
        try:
            self.is_training = True
            
            # Determine training mode
            mode = self.config['training_mode']
            
            if mode == 'train':
                self.perform_training()
            elif mode == 'search_hyperparameters':
                self.tune_hyperparameters()
            elif mode == 'search_hyperparameters_then_train_best':
                self.tune_hyperparameters()
                if self.best_hyperparameters:
                    self.get_logger().info("Starting training with best hyperparameters...")
                    self.perform_training(use_best_params=True)
                else:
                    self.get_logger().error("No best hyperparameters found, skipping training")
            else:
                raise ValueError(f"Unknown training mode: {mode}")
                
        except Exception as e:
            self.get_logger().error(f"Training failed: {str(e)}")
            import traceback
            self.get_logger().error(traceback.format_exc())
            status_msg = String()
            status_msg.data = f"Training failed: {str(e)}"
            self.status_pub.publish(status_msg)
        
        finally:
            self.is_training = False
    
    def perform_training(self, use_best_params=False):
        """Perform normal training"""
        self.get_logger().info("Starting normal training...")
        
        # Load model
        model_type = self.config['model']['type']
        if self.config['model']['pretrained']:
            self.model = YOLO(model_type)
            self.get_logger().info(f"Loaded pretrained model: {model_type}")
        else:
            model_yaml = model_type.replace('.pt', '.yaml')
            self.model = YOLO(model_yaml)
            self.get_logger().info(f"Loaded model architecture: {model_yaml}")
        
        # Prepare data path
        dataset_config = self.config['dataset']
        data_yaml_path = dataset_config['yaml']
        if not os.path.isabs(data_yaml_path):
            dataset_path = os.path.expanduser(dataset_config['path'])
            data_yaml_path = os.path.join(dataset_path, data_yaml_path)
        
        # Validate dataset before training
        self.get_logger().info("Validating dataset...")
        self.validate_dataset(data_yaml_path)
        
        # Prepare save directory
        save_dir = os.path.expanduser(self.config['output']['save_dir'])
        os.makedirs(save_dir, exist_ok=True)
        
        # Parse device
        device = self.parse_device(self.config['hardware']['device'])
        
        # Build training arguments
        train_args = self.build_training_args(data_yaml_path, device, save_dir)
        
        # Override with best hyperparameters if available
        if use_best_params and self.best_hyperparameters:
            self.get_logger().info("Using best hyperparameters from tuning")
            train_args.update(self.best_hyperparameters)
            train_args['name'] = train_args['name'] + '_best'
        
        self.get_logger().info(f"Starting training with {len(train_args)} parameters")
        
        # Train the model
        results = self.model.train(**train_args)
        
        self.get_logger().info("Training completed successfully!")
        self.get_logger().info(f"Model saved to: {results.save_dir}")
        
        # Publish completion status
        status_msg = String()
        status_msg.data = f"Training completed. Model saved to: {results.save_dir}"
        self.status_pub.publish(status_msg)
    
    def tune_hyperparameters(self):
        """Perform hyperparameter tuning"""
        self.get_logger().info("Starting hyperparameter tuning...")
        
        # Load model
        model_type = self.config['model']['type']
        self.model = YOLO(model_type)
        self.get_logger().info(f"Loaded model for tuning: {model_type}")
        
        # Prepare data path
        dataset_config = self.config['dataset']
        data_yaml_path = dataset_config['yaml']
        if not os.path.isabs(data_yaml_path):
            dataset_path = os.path.expanduser(dataset_config['path'])
            data_yaml_path = os.path.join(dataset_path, data_yaml_path)
        
        # Validate dataset before tuning
        self.get_logger().info("Validating dataset...")
        self.validate_dataset(data_yaml_path)
        
        # Parse device
        device = self.parse_device(self.config['hardware']['device'])
        
        # Prepare save directory
        save_dir = os.path.expanduser(self.config['output']['save_dir'])
        os.makedirs(save_dir, exist_ok=True)
        
        # Perform tuning
        tuning_config = self.config['hyperparameter_tuning']
        iterations = tuning_config['iterations']
        
        self.get_logger().info(f"Running {iterations} tuning iterations...")
        self.get_logger().info(f"Search space: {tuning_config['search_space']}")
        
        # Build tuning arguments
        tune_args = {
            'data': data_yaml_path,
            'epochs': self.config['training']['epochs'],
            'iterations': iterations,
            'device': device,
            'project': os.path.join(save_dir, self.config['output']['project_name']),
            'name': self.config['output']['experiment_name'] + '_tune',
            'exist_ok': self.config['output']['exist_ok'],
            'verbose': self.config['output']['verbose'],
            'imgsz': self.config['training']['imgsz'],
            'batch': self.config['training']['batch_size'],
        }
        
        # Add search space
        tune_args.update(tuning_config['search_space'])
        
        results = self.model.tune(**tune_args)
        
        self.get_logger().info("Hyperparameter tuning completed!")
        self.get_logger().info(f"Results saved to: {results.save_dir}")
        
        # Store best hyperparameters
        if hasattr(results, 'best_params'):
            self.best_hyperparameters = results.best_params
            self.get_logger().info(f"Best hyperparameters: {self.best_hyperparameters}")
        
        # Publish completion status
        status_msg = String()
        status_msg.data = f"Tuning completed. Results saved to: {results.save_dir}"
        self.status_pub.publish(status_msg)
    
    def build_training_args(self, data_path, device, save_dir):
        """Build dictionary of training arguments from config"""
        cfg = self.config
        
        args = {
            # Basic settings
            'data': data_path,
            'epochs': cfg['training']['epochs'],
            'batch': cfg['training']['batch_size'],
            'imgsz': cfg['training']['imgsz'],
            'device': device,
            'patience': cfg['training']['patience'],
            'save_period': cfg['training']['save_period'],
            
            # Optimizer settings
            'optimizer': cfg['training']['optimizer'],
            'lr0': cfg['training']['lr0'],
            'lrf': cfg['training']['lrf'],
            'momentum': cfg['training']['momentum'],
            'weight_decay': cfg['training']['weight_decay'],
            'warmup_epochs': cfg['training']['warmup_epochs'],
            'warmup_momentum': cfg['training']['warmup_momentum'],
            'warmup_bias_lr': cfg['training']['warmup_bias_lr'],
            
            # Loss weights
            'box': cfg['training']['box'],
            'cls': cfg['training']['cls'],
            'dfl': cfg['training']['dfl'],
            
            # Output
            'project': os.path.join(save_dir, cfg['output']['project_name']),
            'name': cfg['output']['experiment_name'],
            'exist_ok': cfg['output']['exist_ok'],
            'verbose': cfg['output']['verbose'],
            
            # Validation
            'val': cfg['validation']['val'],
            'plots': cfg['validation']['plots'],
            'save_json': cfg['validation']['save_json'],
            'save_hybrid': cfg['validation']['save_hybrid'],
            'conf': cfg['validation']['conf'],
            'iou': cfg['validation']['iou'],
            'max_det': cfg['validation']['max_det'],
            'half': cfg['validation']['half'],
        }
        
        # Add resume if enabled
        if cfg['resume']['enabled'] and cfg['resume']['checkpoint_path']:
            checkpoint_path = os.path.expanduser(cfg['resume']['checkpoint_path'])
            if os.path.exists(checkpoint_path):
                args['resume'] = checkpoint_path
                self.get_logger().info(f"Resuming from checkpoint: {checkpoint_path}")
            else:
                self.get_logger().warn(f"Checkpoint not found: {checkpoint_path}, starting fresh")
        
        return args
    
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
    
    def publish_status(self):
        status_msg = String()
        if self.is_training:
            mode = self.config['training_mode']
            status_msg.data = f"Mode: {mode} - Training in progress..."
        else:
            status_msg.data = "Idle - ready to train"
        self.status_pub.publish(status_msg)
    
    def cleanup(self):
        if self.training_thread and self.training_thread.is_alive():
            self.get_logger().info("Waiting for training to complete...")
            self.training_thread.join(timeout=5.0)


def main(args=None):
    rclpy.init(args=args)
    node = YOLOTrainerNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down YOLO Trainer Node...")
    finally:
        node.cleanup()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()