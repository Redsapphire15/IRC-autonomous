from ultralytics import YOLO
import cv2
from ultralytics.utils.plotting import Annotator  # ultralytics.yolo.utils.plotting is deprecated

model = YOLO('/home/kavin/Downloads/arrowspt2/runs/detect/train7/weights/best.pt')
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

while True:
    _, img = cap.read()
    
    # BGR to RGB conversion is performed under the hood
    # see: https://github.com/ultralytics/ultralytics/issues/2575
    results = model.predict(img)

    arrow_in_center = False

    for r in results:
        annotator = Annotator(img)
        boxes = r.boxes
        for box in boxes:
            b = box.xyxy[0]  # get box coordinates in (left, top, right, bottom) format
            c = box.cls
            annotator.box_label(b, model.names[int(c)])

            # Get depth data for the bounding box
            left, top, right, bottom = map(int, b)
            arrow_center = (left+right)/2
            #depth = depth_frame.get_distance((left + right) // 2, (top + bottom) // 2)
            #print(f"Depth for box {b}: {depth} meters")

            # Check if the arrow is in the center along the x-axis
            img_center_x = img.shape[1] // 2  # Calculate the center along x-axis
            # if left <= img_center_x <= right:
            if abs(arrow_center-img_center_x)<20:
                arrow_in_center = True

    if arrow_in_center:
        cv2.putText(img, "Arrow in centre", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    img = annotator.result()  
    cv2.imshow('YOLO V8 Detection', img)     
    if cv2.waitKey(1) & 0xFF == ord(' '):
        break

cap.release()
cv2.destroyAllWindows()