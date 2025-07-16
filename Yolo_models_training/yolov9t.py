%cd /home/hasan/drone_p3_implementation/


from ultralytics import YOLO

# Build a YOLOv9c model from scratch
#model = YOLO("yolov9t.yaml")

# Build a YOLOv9c model from pretrained weight
model = YOLO("yolov9t.pt")

# Display model information (optional)
model.info()

# Train the model on the COCO8 example dataset for 100 epochs
results = model.train(data="/home/hasan/drone_p3_implementation/drone3_latest-2/data.yaml" ,epochs=200 ,imgsz=640 ,plots=True, device=0)

# Run inference with the YOLOv9c model on the 'bus.jpg' image
#results = model("path/to/bus.jpg")


#%%

!yolo task=detect \
mode=val \
model="/home/hasan/drone_p3_implementation/runs/detect/yolov9t_train/weights/best.pt" \
data="/home/hasan/drone_p3_implementation/drone3_latest-2/data1.yaml"  \
device=0


#%%


%cd /home/hasan/drone_p3_implementation/

!yolo task=detect mode=predict model="/home/hasan/drone_p3_implementation/runs/detect/yolov9t_train/weights/best.pt" conf=0.5 source='/home/hasan/drone_p3_implementation/drone3_latest-2/test/images/' save=True device=0
