from ultralytics import YOLO
import cv2
import numpy as np
import pyrealsense2 as rs
from ultralytics.utils.plotting import Annotator

#model = YOLO('/home/kavin/Downloads/best.pt')
model = YOLO('/home/kavin/Downloads/best.pt')

# Configure RealSense pipeline to get color and depth frames
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

# Start streaming
pipeline.start(config)

while True:
    # Wait for coherent color and depth frames
    frames = pipeline.wait_for_frames()
    color_frame = frames.get_color_frame()
    depth_frame = frames.get_depth_frame()
    if not color_frame or not depth_frame:
        continue

    # Convert RealSense color frame to OpenCV format
    img = np.asanyarray(color_frame.get_data())


    # Run YOLO model inference
    results = model.predict(img, conf = 0.5, max_det = 2)

    arrow_in_center = False

    if results == None:
        arrow_detected = False
    else:
        arrow_detected = True

    for r in results:
        print("result:" + str(len(results)))
        if len(results)>=2:
            break
            
        annotator = Annotator(img)
        boxes = r.boxes
        depths = []
        for box in boxes:
            b = box.xyxy[0]  # get box coordinates in (left, top, right, bottom) format
            c = box.cls
            annotator.box_label(b, model.names[int(c)])
            #print(f"Depth for box {b}: {depth} meters")   

            # Get depth data for the bounding box
            left, top, right, bottom = map(int, b)
            arrow_center = (left+right)/2
            depth = depth_frame.get_distance((left + right) // 2, (top + bottom) // 2)
            depths.append(depth)
            cv2.rectangle(img, (left, top), (right, bottom), (0, 255, 0), 2)

            # Check if the arrow is in the center along the x-axis
            img_center_x = img.shape[1] // 2  # Calculate the center along x-axis
            # if left <= img_center_x <= right:
            print(arrow_center-img_center_x)
            if abs(arrow_center-img_center_x)<100:
                arrow_in_center = True
            print(f"Depth for box {b}: {depth} meters")   
    if arrow_in_center:
        cv2.putText(img, "Arrow in centre", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow('YOLO V8 Detection', img)

    if cv2.waitKey(1) & 0xFF == ord(' '):
        break

# Stop streaming
pipeline.stop()
cv2.destroyAllWindows()
