import cv2
from ultralytics import YOLO
import numpy as np
import time

cap = cv2.VideoCapture("rawvideo.mp4")

model = YOLO(r"weights/cell_detection_YOLO.pt")

pTime = 0

while(True):
    ret,frame=cap.read()

    results = model(frame,device='cpu')
    result=results[0]

    bboxes = np.array(result.boxes.xyxy.cpu(),dtype='int')
    classes=np.array(result.boxes.cls.cpu(),dtype='int')

    for clss,bbox in zip(classes,bboxes):
        x,y,x2,y2 = bbox
        cv2.rectangle(frame,(x,y),(x2,y2),(0,0,255),2)
    start_time = time.time()
    elapsed_time = start_time - pTime 
    fps = 1 / elapsed_time
    pTime=start_time
    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    cv2.imshow("Vid",frame)

    key=cv2.waitKey(1)
    
    if key==27:
        break

cap.release()
cv2.destroyAllWindows()
