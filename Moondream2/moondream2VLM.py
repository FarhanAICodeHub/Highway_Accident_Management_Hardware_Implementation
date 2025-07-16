# Install required packages if not already installed
# !pip install transformers einops

from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image
import torch  # Import PyTorch 
import os

# Set paths
image_folder = "/home/hasan/drone_p2_implementation/det_video_frames/" # Update this to your folder path
image_paths = [os.path.join(image_folder, img) for img in os.listdir(image_folder) if img.endswith(".jpg")]

# Load the model and tokenizer
model_id = "vikhyatk/moondream2"
revision = "2024-08-26"
model = AutoModelForCausalLM.from_pretrained(
    model_id, trust_remote_code=True, revision=revision
)
tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)

# Check for GPU availability and move the model to the appropriate device
device = "cuda:0" #if torch.cuda.is_available() else "cpu"
model.to(device)

 

def generate_caption(image_path, prompt="""Describe the accident scene in the image, including the vehicles involved, 
                         visible damages. Provide a direct, detailed summary(2 lines) of the accident without referencing any time stamp or bounding boxes.
                          """):

 
 
    image = Image.open(image_path)
    
    # Encoding the image with the model and moving it to the device (GPU or CPU)
    enc_image = model.encode_image(image).to(device)  # Ensure the encoded image is on the correct device
    
    # Generate the description based on the prompt
    description = model.answer_question(enc_image, prompt, tokenizer)
    return description

 
# Create or open a text file to save descriptions
output_file = "image_descriptions1.txt"
with open(output_file, "w") as file:
    # Generate descriptions for each image and save them to the file with numbered captions
    for i, image_path in enumerate(image_paths, start=1):
        caption = generate_caption(image_path)
        file.write(f"{i}: {caption}\n")
        print(f"{i}: {caption}")

print(f"\nAll descriptions have been saved to {output_file}")
