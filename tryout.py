import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, Label, Button, StringVar, Entry
from PIL import Image, ImageTk
from diffusers import StableDiffusionInpaintPipeline
import torch
#from sam2.sam2.build_sam import build_sam2_video_predictor

from sam2.sam2.build_sam import build_sam2_video_predictor

sam2_checkpoint = "../checkpoints/sam2.1_hiera_tiny.pt"
model_cfg = "sam2_hiera_l.yaml"

predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint)


# Load Stable Diffusion Inpainting model
device = "cuda" if torch.cuda.is_available() else "cpu"
pipe = StableDiffusionInpaintPipeline.from_pretrained("runwayml/stable-diffusion-inpainting").to(device)

def generate_image_with_mask(prompt, frame, mask):
    """Generate an image using Stable Diffusion Inpainting based on a prompt and mask."""
    mask = (mask * 255).astype(np.uint8)  # Ensure mask is binary
    mask_pil = Image.fromarray(mask).convert("L")
    frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    
    inpainted_image = pipe(prompt=prompt, image=frame_pil, mask_image=mask_pil).images[0]
    return inpainted_image

def process_video_with_sam2(video_path, prompt):
    """Segment objects with SAM2, generate replacements using Stable Diffusion, and process video."""
    cap = cv2.VideoCapture(video_path)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    out = cv2.VideoWriter('output_sam2.avi', cv2.VideoWriter_fourcc(*'XVID'), 30, (frame_width, frame_height))
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Generate masks for the current frame using SAM2
        masks = mask_generator.generate(frame)
        
        if masks:
            largest_mask = max(masks, key=lambda x: np.sum(x['segmentation']))
            mask = largest_mask['segmentation']
            
            inpainted_image = generate_image_with_mask(prompt, frame, mask)
            
            mask_3d = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
            inpainted_array = np.array(inpainted_image)
            
            frame[mask_3d == 255] = inpainted_array[mask_3d == 255]
        
        out.write(frame)
    
    cap.release()
    out.release()
    print("Video processing completed. Saved as output_sam2.avi")

def open_video():
    """Open a video file."""
    file_path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4;*.avi")])
    video_path.set(file_path)

def start_processing():
    """Segment and process video using SAM2 and Stable Diffusion."""
    prompt = prompt_text.get()
    video_file = video_path.get()
    
    if not prompt or not video_file:
        result_label.config(text="Please provide both a video and a prompt.")
        return
    
    result_label.config(text="Processing video...")
    process_video_with_sam2(video_file, prompt)
    result_label.config(text="Video processed successfully!")

# Create Tkinter UI
root = tk.Tk()
root.title("SAM2 + Stable Diffusion Video Processor")

video_path = StringVar()
prompt_text = StringVar()

Label(root, text="Select Video:").pack()
Button(root, text="Browse", command=open_video).pack()
Label(root, textvariable=video_path).pack()

Label(root, text="Enter Prompt:").pack()
Entry(root, textvariable=prompt_text).pack()

Button(root, text="Start", command=start_processing).pack()
result_label = Label(root, text="")
result_label.pack()

root.mainloop()


