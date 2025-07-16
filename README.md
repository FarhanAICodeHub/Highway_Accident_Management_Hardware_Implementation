# Multi-Modal Autonomous Drone System for Real-Time Highway Incident Detection and Analysis

## Usage Instructions

### **Prerequisites**

* Python 3.8+ environment
* NVIDIA GPU (recommended for YOLO and LLaVA models)
* [Ultralytics YOLO](https://docs.ultralytics.com/)
* [Transformers (HuggingFace)](https://huggingface.co/docs/transformers/index)
* DJI SDK/Cloud API or simulation mode
* OpenAI GPT-4o API key

This repo is related to implementation of highway accident management system that was implemented using hardware




### **Quick Start**

1. **Clone the Repository**

   ```sh
   git clone  https://github.com/afaq005/HighwayAccident Management Hardware Implementation.git
   cd autonomous-highway-incident
   ```

2. **Configure Your Environment**

   * Download YOLOv12n weights and LLaVA-OneVision-Qwen2 models as described in the scripts.
   * Set your OpenAI API key for GPT-4o mini  as an environment variable.
   * Optionally, configure DJI drone connection details or set to simulation (test video) mode.

3. **Run the Pipeline**

   ```sh
   python complete_integration_pipeline.py
   ```

   * The script will:

     * Extract coordinates from transcript (using LLM)
     * Deploy the drone or use test video
     * Run YOLOv12n accident detection, save frames
     * Generate VLM scene descriptions
     * Summarize alerts with GPT-4o mini

4. **Outputs**

   * Scene descriptions: `image_descriptions.txt`
   * Alert summaries: `collective_summary.txt`
   * Detected frames: `accident_frames/`

---

## Example Outputs

* **YOLOv12n Detection:**
  ![YOLOv12n Example](170ce7a7-cc93-4526-a90e-df2a8073658d.png)

* **VLM Descriptions:**

  * Scenario 1: “The image shows a car accident with a blue bounding box... high risk of being involved in an accident, with a probability of 83%...”
  * Scenario 2: “The image shows a scene with three vehicles... accident 0.56... with bystanders present...”

* **LLM Alert (for rescue center):**

  * “Accident Scene Summary: High. A car accident has occurred near a stadium, with a significant risk factor indicated by the high probability of 83%.”
  * “Deploy emergency medical services immediately for potential injuries and ensure police are on-site to manage traffic and secure the area.”

---

## How to Adapt or Extend

* Swap YOLOv12n for any YOLO model you prefer (with a simple config change).
* Integrate with live DJI drone using provided Python interface or REST API.
* Fine-tune VLM/LLM components for new domains (e.g., fire, disaster, search and rescue).
* Build a web dashboard or REST API for real-time alert delivery.

---


## Key Components

### 1. **DJI Matrice 30T Integration**

* Fully autonomous takeoff, navigation, and landing via SDK.
* Route planning to GPS coordinates provided by LLM extraction.
* Live RTSP video streaming, compatible with both real drone and test video modes.

### 2. **YOLOv12n Object Detection**

* Fine-tuned on custom incident datasets for robust highway accident/fire detection.
* Supports real-time inference and high-confidence frame filtering.
* Automatically saves key frames for downstream analysis.

### 3. **LLaVA-OneVision-Qwen2 (VLM)**

* Converts key images into rich, human-readable descriptions.
* Captures visual context, accident risk, bystanders, and environmental cues.

### 4. **GPT-4o mini (LLM)**

* Extracts coordinates from incident transcripts.
* Summarizes VLM outputs to generate alert-level summaries and recommended emergency actions.
* Designed for structured, actionable outputs compatible with rescue center needs.

---



## Framework

![figm](https://github.com/user-attachments/assets/95153832-18b1-4293-a6ff-f332715b9288)

## Hardware Implementation

![fig_hardware](https://github.com/user-attachments/assets/ce5a3017-a044-45e9-b4bf-6ff468b71efc)


##### Download the dataset from dataset.py
##### All the files related to yolo variants are listed above, install ultralytics latest version before usage
##### Yolov12n model pretrained weights are uploaded


 **comprehensive README.md**---

# Autonomous Drone-based Highway Incident Management System

## Overview

This project presents a fully autonomous highway accident management system powered by a DJI Matrice 30T drone, advanced object detection, and multi-modal AI for **real-time accident detection, analysis, and emergency alert generation**. The system fuses:

* **DJI Matrice 30T** for aerial incident scene acquisition and autonomous navigation.
* **YOLOv12n** for robust, real-time detection of accidents (e.g., collisions, fires).
* **LLaVA-OneVision-Qwen2** (VLM) for visual scene interpretation and rich description generation.
* **GPT-4o mini** (LLM) for automated, concise alerting and emergency action recommendation.

This solution aims to reduce incident response times and maximize situational awareness for highway safety teams and first responders.

---

## Abstract

> Highway accidents remain a critical public safety challenge, where delayed response times intensify injury severity and fatalities. Traditional systems relying on manual reporting suffer from inefficiencies and critical delays during emergencies. This paper presents an autonomous drone-based highway incident management system that integrates advanced aerial technology with Vision-Language Models (VLMs) and Large Language Models (LLMs) to enable rapid detection, analysis, and response.
>
> The framework employs a DJI Matrice 30T drone equipped with a dock station for autonomous deployment for incident analysis by yielding incident coordinates provided by authorities. Upon arrival, real-time video is analyzed using an optimized fine-tuned YOLOv12n model selected after rigorous comparison of nine YOLO variants to detect incidents such as collisions and fires with high precision.
>
> To generate actionable intelligence, high-confidence video frames are processed using a hybrid vision-language pipeline comprising of LLaVA-OneVision-Qwen2 and GPT-4o API. This combination yields detailed natural language descriptions and structured summaries of incident scenes, offering enhanced contextual awareness of the incident site. By combining autonomous drone navigation, real-time object detection, VLM-based scene interpretation, and LLM-driven summarization, this end-to-end solution enhances situational awareness and decision-making for emergency responders. These results demonstrate the practical value of integrating multi-modal AI with UAVs in critical safety domains, laying the groundwork for scalable, intelligent transportation infrastructure and next-generation emergency response systems.

---

## System Workflow

The pipeline consists of the following stages (see attached figures):

1. **Incident Scenario Input**
   Incident reports or highway authorities provide a transcript or notification (possibly containing GPS coordinates).

2. **Coordinate Extraction**
   The system uses a large language model (GPT-4o mini) to extract precise latitude and longitude from the transcript.

3. **Autonomous Drone Deployment**

   * The DJI Matrice 30T drone, with docking station, is automatically dispatched to the incident coordinates.
   * The route planner calculates the optimal path and initiates drone takeoff.
   * (Optionally: in simulation mode, a pre-recorded video can be used.)

4. **Live Video Feedback & Analysis**

   * The drone streams real-time video of the scene back to the server.
   * This feed is processed by a fine-tuned YOLOv12n model, which was selected after comparison with nine other YOLO variants, for reliable detection of accidents, collisions, or fires.

5. **Frame Selection & VLM Description**

   * Frames with high-confidence accident detections are saved.
   * Each frame is passed through LLaVA-OneVision-Qwen2 (Vision-Language Model), generating detailed natural language descriptions that summarize the scene context, accident type, and observed risks.

6. **LLM-driven Summarization & Alerting**

   * All generated scene descriptions are collectively summarized using GPT-4o mini.
   * The output is a concise, actionable incident summary and suggested emergency response (e.g., deploy ambulance, secure area), ready for dispatch to rescue teams or traffic management centers.

 
 
 

## Contact

* \[Afaq@jbnu.ac.kr], \[Jeonbuk Nationanl University, South Korea]


---

**This project demonstrates the practical value of multi-modal AI for real-world safety-critical infrastructure. For feedback or collaboration, feel free to open an issue or contact the author.**

