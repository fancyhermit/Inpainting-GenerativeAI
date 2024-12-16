# Import necessary libraries
import cv2
from ultralytics import SAM  # Make sure ultralytics is installed

# Load the SAM model (replace 'sam2.1_b.pt' with the correct path to your model file)
model = SAM('mobile_sam.pt')

model(source=0,show=True)