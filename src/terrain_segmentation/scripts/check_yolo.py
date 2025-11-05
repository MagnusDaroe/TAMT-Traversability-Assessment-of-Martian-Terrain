import torch
import argparse
import shutil
from pathlib import Path
from ultralytics import YOLO
import warnings





model_type = 'yolo11n-seg.pt'  # Use base architecture
new_model = YOLO(model_type)

print(new_model)