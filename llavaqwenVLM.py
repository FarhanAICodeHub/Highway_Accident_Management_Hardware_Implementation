import os
import torch
from PIL import Image
from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration

# Set the model ID and specify the GPU  
model_id = "llava-hf/llava-onevision-qwen2-0.5b-ov-hf"
model = LlavaOnevisionForConditionalGeneration.from_pretrained(
    model_id, 
    torch_dtype=torch.float16, 
    low_cpu_mem_usage=True, 
).to('cuda:1')

processor = AutoProcessor.from_pretrained(model_id)

# Define the folder where your images are located
image_folder = "/home/hasan/drone_p3_implementation/accident_frames/"
output_file = "image_descriptions.txt"  # Output file to save descriptions

# Define a chat history and prompt template (Remove the question)
conversation = [
    {
        "role": "user",
        "content": [
            {"type": "image"},  # Only send image as input, no question prompt
        ],
    },
]
prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)

# Open the output file in write mode
with open(output_file, 'w') as f:
    # Loop through the images in the folder (taking first 10 images)
    for idx, image_filename in enumerate(os.listdir(image_folder)[:10]):
        # Ensure it's an image file (you can filter for .jpg, .png, etc.)
        if image_filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            # Construct the full image path
            image_path = os.path.join(image_folder, image_filename)
            
            # Open the image
            raw_image = Image.open(image_path)
            
            # Process the image and prompt
            inputs = processor(images=raw_image, text=prompt, return_tensors='pt').to('cuda:1', torch.float16)
            
            # Generate the description
            output = model.generate(**inputs, max_new_tokens=100, do_sample=False)
            
            # Decode and remove the question/prompt from the output
            description = processor.decode(output[0][2:], skip_special_tokens=True)
            
            # Clean the output by removing the unwanted word "assistant"
            description_cleaned = description.replace("assistant", "").strip()
            
            # Write the description to the file
            f.write(f"Image description {idx+1}:\n")
            f.write(description_cleaned + "\n\n")
            
            # Optionally, print the cleaned description
            print(f"Image description {idx+1}:")
            print(description_cleaned)
            print("-" * 50)
