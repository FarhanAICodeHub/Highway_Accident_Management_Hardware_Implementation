%cd /home/hasan/drone_p3_implementation/


from ultralytics import YOLO

# Build a YOLOv10n model from scratch
#model = YOLO("yolov10n.yaml")

# Build a YOLOv10n model from pretrained  weights
model = YOLO("yolov10n.pt")

# Display model information (optional)
model.info()

# Train the model on the COCO8 example dataset for 100 epochs
results = model.train(data="/home/hasan/drone_p3_implementation/drone3_latest-2/data.yaml" ,epochs=200 ,imgsz=640 ,plots=True, device=0)

# Run inference with the YOLOv10n model on the 'bus.jpg' image
#results = model("path/to/bus.jpg")

#%%

!yolo task=detect \
mode=val \
model="/home/hasan/drone_p3_implementation/runs/detect/yolov10n_train/weights/best.pt" \
data="/home/hasan/drone_p3_implementation/drone3_latest-2/data1.yaml"  \
device=0

#%%


!yolo task=detect \
mode=val \
model="/home/hasan/drone_p3_implementation/runs/detect/yolov10n_train/weights/best.pt" \
data="/home/hasan/drone_p3_implementation/drone3_latest-2/data1.yaml"  \
device=0
