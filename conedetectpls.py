from ultralytics import YOLO
import cv2
import numpy as np
import pyrealsense2 as rs
from ultralytics.utils.plotting import Annotator

# Load YOLO model
model = YOLO('/home/kavin/Downloads/Safety_Cone_detection-main/Models/Cone.pt')  # Replace with the path to your YOLO model

# Configure RealSense pipeline to get color frames
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
pipeline.start(config)

while True:
    # Wait for coherent color frame
    frames = pipeline.wait_for_frames()
    color_frame = frames.get_color_frame()
    if not color_frame:
        continue

    # Convert RealSense color frame to OpenCV format
    img = np.asanyarray(color_frame.get_data())

    # Run YOLO model inference
    results = model.predict(img, conf=0.75, max_det=5)

    for r in results:
        annotator = Annotator(img)
        boxes = r.boxes
        for box in boxes:
            b = box.xyxy[0]  # get box coordinates in (left, top, right, bottom) format
            c = box.cls
            annotator.box_label(b, model.names[int(c)])

            # Draw rectangle around the cone
            left, top, right, bottom = map(int, b)
            cv2.rectangle(img, (left, top), (right, bottom), (0, 255, 0), 2)

    cv2.imshow('YOLO Cone Detection', img)

    if cv2.waitKey(1) & 0xFF == ord(' '):
        break

# Stop streaming
pipeline.stop()
cv2.destroyAllWindows()
