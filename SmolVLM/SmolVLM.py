from transformers import AutoProcessor, AutoModelForVision2Seq
import torch
from PIL import Image
import os

# Set device gpu/cpu
DEVICE = "cuda:1" if torch.cuda.is_available() else "cpu"

# Load model and processor
processor = AutoProcessor.from_pretrained("HuggingFaceTB/SmolVLM-Instruct")
model = AutoModelForVision2Seq.from_pretrained(
    "HuggingFaceTB/SmolVLM-Instruct",
    torch_dtype=torch.bfloat16,
    _attn_implementation="flash_attention_2" if DEVICE == "cuda" else "eager"
).to(DEVICE)

# Folder containing images
image_folder = "/home/hasan/drone_p3_implementation/accident_frames/"
image_files = sorted([os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.endswith((".jpg", ".png"))])[:10]  # Select first 10 image 

# Output file to save descriptions
output_file = "/home/hasan/drone_p3_implementation/accident_descriptions.txt"

# Process each image one by one
with open(output_file, "w") as file:
    for idx, image_path in enumerate(image_files):
        image = Image.open(image_path)

        # Prepare inputs without including the question in the output
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"}, 
                    {"type": "text", "text": """Describe the accident scene in the image, including the vehicles involved,   
                         visible damages. """}
                ]
            },
        ]

        # Prepare inputs
        prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = processor(text=prompt, images=[image], return_tensors="pt").to(DEVICE)

        # Generate output
        generated_ids = model.generate(**inputs, max_new_tokens=100)
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

        # Remove unwanted parts like "User:<image>Can you describe..."
        if "Assistant:" in generated_text:
            generated_text = generated_text.split("Assistant:")[-1].strip()

        # Print results
        description = f"Image {idx + 1} description: {generated_text}\n"
        print(description)

        # Save to file
        file.write(description + "\n")
