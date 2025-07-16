#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
 
@author: hasan
"""


import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
import torch

#%%


import os
import shutil
import time
import cv2
import torch
from PIL import Image


#%%

# ========== CONFIGURATION =============

# --- Use this to switch between "test_video" or "drone" mode ---
USE_TEST_VIDEO = True  # Set to False for RTSP/Drone mode

# Test video path for offline mode
TEST_VIDEO_PATH = "/home/hasan/drone_p3_implementation/two_det_videos/v7e_orig.mp4" #"/path/to/sample_accident_video.mp4"  # CHANGE THIS

# RTSP stream for drone
RTSP_URL = "rtsp://user:password@192.168.23.166:8554/streaming/live/1"  # CHANGE THIS

# Accident frames folder (all saved images go here)
ACCIDENT_FRAMES_FOLDER = "accident_frames"

# Model paths
YOLO_WEIGHTS = "/home/hasan/drone_p3_implementation/runs/detect/yolov12n_train/weights/best.pt" #"yolov12/best.pt"  # or yolov12n/best.pt, whatever you have
VLM_MODEL_ID = "llava-hf/llava-onevision-qwen2-0.5b-ov-hf" 

# ========== 1. Extract Coordinates with GPT-4o mini ==============

def extract_coordinates_gpt4o(transcript, api_key):
    import openai
    openai.api_key = "api_key"
#api_key
    prompt = (
        "Extract the latitude and longitude coordinates from the following accident transcript. "
        "Only return coordinates as two comma-separated floats (e.g., 31.12345, 74.56789):\n"
        f"{transcript}"
    )
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=30,
            temperature=0
        )
        text = response.choices[0].message["content"]
        parts = text.strip().split(",")
        lat, lon = float(parts[0]), float(parts[1])
        print(f"Coordinates extracted: {lat}, {lon}")
        return lat, lon
    except Exception as e:
        print(f"Error extracting coordinates: {e}")
        raise

# ========== 2. Run YOLO, Save Accident Frames =============

def clean_folder(folder):
    if os.path.exists(folder):
        shutil.rmtree(folder)
    os.makedirs(folder)

def detect_accident_and_save_frames(source, yolo_weights, output_folder, device="cuda", max_frames=10):
    from ultralytics import YOLO
    model = YOLO(yolo_weights)#.to(device)

    clean_folder(output_folder)

    if isinstance(source, str) and os.path.isfile(source):
        cap = cv2.VideoCapture(source)  # test video
    else:
        cap = cv2.VideoCapture(source)  # RTSP or video file

    if not cap.isOpened():
        print("Error: Cannot open video source.")
        return []

    frame_count = 0
    saved_frames = 0
    accident_frames = []

    while cap.isOpened() and saved_frames < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)
        for box in results[0].boxes.data.tolist():
            # Assuming class 0 is 'accident' - change as needed!
            class_id = int(box[5])
            confidence = box[4]
            if class_id == 0 and confidence > 0.5:
                frame_path = os.path.join(output_folder, f"accident_{saved_frames + 1}.jpg")
                cv2.imwrite(frame_path, frame)
                accident_frames.append(frame_path)
                print(f"Saved accident frame: {frame_path}")
                saved_frames += 1
                break  # Save only one detection per frame

        frame_count += 1
    cap.release()
    return accident_frames

# ========== 3. VLM Inference for Image Descriptions ===========

def describe_images_with_llava(image_folder, vlm_model_id, device="cuda", out_txt="image_descriptions.txt", max_imgs=10):
    from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration
    model = LlavaOnevisionForConditionalGeneration.from_pretrained(
        vlm_model_id, torch_dtype=torch.float16, low_cpu_mem_usage=True).to(device)
    processor = AutoProcessor.from_pretrained(vlm_model_id)

    conversation = [{"role": "user", "content": [{"type": "image"}]}]
    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)

    image_files = sorted([
        f for f in os.listdir(image_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ])[:max_imgs]
    descriptions = []

    with open(out_txt, 'w') as f:
        for idx, img_file in enumerate(image_files):
            img_path = os.path.join(image_folder, img_file)
            raw_image = Image.open(img_path)
            inputs = processor(images=raw_image, text=prompt, return_tensors='pt').to(device, torch.float16)
            output = model.generate(**inputs, max_new_tokens=100, do_sample=False)
            description = processor.decode(output[0][2:], skip_special_tokens=True)
            description_cleaned = description.replace("assistant", "").strip()
            f.write(f"Image description {idx+1}:\n")
            f.write(description_cleaned + "\n\n")
            descriptions.append(description_cleaned)
            print(f"Image description {idx+1}: {description_cleaned}")
            print("-" * 40)
    return descriptions

# ========== 4. GPT-4o mini Summarization ==============

def summarize_accident_with_gpt4o(descriptions, api_key, out_txt="collective_summary.txt"):
    import openai
    openai.api_key = "api_key"
#api_key
    descriptions_text = "\n".join(descriptions)
    prompt = (
        "You are an expert accident response supervisor. "
        "You'll be provided accident descriptions (around 10, same accident). "
        "First, describe the accident scene in two lines (with alert level and summary), then "
        "guide the team for suggested health/emergency responses (max two lines, these can be related to rescue, ambulance, disaster or police team).\n"
        "Description text from accident is:\n\n" + descriptions_text
    )
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200, temperature=0.3, top_p=1, frequency_penalty=0.2, presence_penalty=0.2
        )
        summary = response.choices[0].message["content"]
        with open(out_txt, "w") as f:
            f.write("Collective Summary:\n" + summary)
        print(f"Collective summary saved to {out_txt}")
        return summary.strip()
    except Exception as e:
        print(f"Error summarizing text: {e}")
        return None

def fly_dji_m30t_to_coords(lat, lon, altitude=30.0):
    """
    Commands the DJI M30T to take off and fly to the specified GPS coordinates at a given altitude.
    Note: You must have the DJI OSDK/Cloud API running, with a Python bridge or REST proxy.

    :param lat: Target latitude (float)
    :param lon: Target longitude (float)
    :param altitude: Target altitude in meters (default: 30.0)
    """
    try:
        # Example using a hypothetical Python SDK wrapper (replace with actual implementation!)
        from dji_sdk import M30T

        print(f"[DJI] Connecting to DJI M30T drone...")
        drone = M30T.connect(serial_port="/dev/ttyUSB0")  # Adjust your serial port

        print("[DJI] Arming drone...")
        drone.arm()

        print("[DJI] Taking off...")
        drone.takeoff()

        print(f"[DJI] Flying to GPS coordinates: {lat}, {lon}, altitude: {altitude}m...")
        drone.go_to_gps(latitude=lat, longitude=lon, altitude=altitude)

        print("[DJI] Arrived at target location.")

        # Optional: start video streaming, etc.
        # drone.start_video_stream()

        return True

    except Exception as e:
        print(f"[DJI ERROR] Failed to fly M30T: {e}")
        return False

# ========== 6. MAIN PIPELINE =================

def main():
    # ==== USER INPUTS ====
    transcript = "There is an accident at location 31.123456, 74.123456 near the stadium."  # CHANGE as needed!
    gpt4o_api_key = "api_key"
#"sk-xxxxxxxx"  # PUT YOUR KEY

    # 1. Extract coordinates from transcript with GPT-4o
    lat, lon = extract_coordinates_gpt4o(transcript, gpt4o_api_key)

    # 2. If drone, move to GPS coords (if not, just skip)
    if not USE_TEST_VIDEO:
        fly_dji_m30t_to_coords(lat, lon)
        video_source = RTSP_URL
    else:
        video_source = TEST_VIDEO_PATH

    #accident_frames="/home/hasan/drone_p3_implementation/accident_frames1/"
    # 3. Run YOLO accident detection and save images
    accident_frames = detect_accident_and_save_frames(
        video_source, YOLO_WEIGHTS, ACCIDENT_FRAMES_FOLDER, device="cuda"
    )
    if not accident_frames:
        print("No accident frames detected.")
        return

    # 4. VLM image description (LLaVA Qwen2)
    image_descriptions = describe_images_with_llava(
        ACCIDENT_FRAMES_FOLDER, VLM_MODEL_ID, device="cuda", out_txt="image_descriptions.txt"
    )

    # 5. Summarize everything with GPT-4o mini
    summary = summarize_accident_with_gpt4o(image_descriptions, gpt4o_api_key, out_txt="collective_summary.txt")
    print("----- FINAL SUMMARY -----")
    print(summary)

if __name__ == "__main__":
    main()

# ====================

# Instructions:
# - Set USE_TEST_VIDEO = False for live drone/RTSP stream.
# - Replace model paths, API keys, and file paths for your system.
# - Place this script at the root of your project and run with: python script_name.py
# - For drone integration, insert DJI SDK logic into fly_dji_m30t_to_coords.
