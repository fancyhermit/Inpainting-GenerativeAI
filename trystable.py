import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, Label, Button, StringVar, Entry
from PIL import Image, ImageTk
from diffusers import StableDiffusionPipeline
import torch

# Load Stable Diffusion model
device = "cuda" if torch.cuda.is_available() else "cpu"
pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5").to(device)

def generate_image_from_prompt(prompt):
    """Generate an image based on the user's prompt using Stable Diffusion."""
    image = pipe(prompt).images[0]
    image.save('generated_object.png')
    return image

def process_video(video_path, replacement_image):
    """Replace an object in the video with a generated image."""
    cap = cv2.VideoCapture(video_path)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    out = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc(*'XVID'), 30, (frame_width, frame_height))
    
    replacement_image = cv2.cvtColor(np.array(replacement_image), cv2.COLOR_RGB2BGR)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Dummy bounding box: replace detected object
        x, y, w, h = 100, 100, 200, 200  # Replace this with real object detection logic
        
        # Resize and overlay replacement image
        resized_replacement = cv2.resize(replacement_image, (w, h))
        frame[y:y+h, x:x+w] = resized_replacement
        
        out.write(frame)
    
    cap.release()
    out.release()
    print("Video processing completed. Saved as output.avi")

def open_video():
    """Open a video file."""
    file_path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4;*.avi")])
    video_path.set(file_path)

def start_processing():
    """Generate image and process video."""
    prompt = prompt_text.get()
    video_file = video_path.get()
    
    if not prompt or not video_file:
        result_label.config(text="Please provide both a video and a prompt.")
        return
    
    result_label.config(text="Generating image...")
    generated_image = generate_image_from_prompt(prompt)
    generated_image.show()
    
    result_label.config(text="Processing video...")
    process_video(video_file, generated_image)
    result_label.config(text="Video processed successfully!")

# Create Tkinter UI
root = tk.Tk()
root.title("Object Replacement with Stable Diffusion")

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
