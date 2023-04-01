# Cell dent detection

## Introduction

## Data
With the help of Roboflow we uploaded our video datasets, augmented and preprocessed our data. The platform also supports integration with popular deep learning frameworks like TensorFlow, PyTorch, and Keras, as well as export to common formats like ONNX and TensorFlow Lite for deployment on various platforms. 
Roboflow uses a process called video frame extraction to convert a video file into a set of individual image files. This process involves taking each frame of the video and saving it as a separate image file. The frames are then saved as individual image files, with a filename that includes the frame number and file extension (e.g., frame_001.jpg, frame_002.jpg, etc.).
*	Uploaded 40 slow-motion videos of dented batteries to Roboflow to collect data from different angles.
* Used Roboflow to extract individual image frames from the videos and preprocess the data.
*	Saved 5 frames per second, resulting in 4000 images.
*	Applied Auto-Orient and Resize techniques to preprocess the images (640x640).
*	Annotated dent images to divide them into different classes and marked others as null.
*	Split the dataset into train, validation, and test sets in an 80:10:10 ratio.
*	Generated augmented versions of each image in the training set to create new training examples for the model to learn from.
*	Augmentation techniques applied include Horizontal and Vertical Flips, Rotation between -15° and +15°, and Exposure adjustment of the bounding box between -20% and +20%.

Annotation examples:
![image](https://user-images.githubusercontent.com/96420770/229296226-65412aa5-09f7-4c19-be93-a7685f372910.png)
![image](https://user-images.githubusercontent.com/96420770/229296232-d0139473-ce95-4967-a4e3-aaf8c29db80d.png)

## Model

## Usage
* Batteries with dents can be dangerous as they can expand and/or leak. This could damage electronic components. This battery cell dent detection model can be deployed using Arduino and RaspberryPi to detect such dented battery cells. 
* This has wide applications in electric car manufacturing where machines can be used to filter out unhealthy cells. This can help to prevent safety issues such as thermal runaway, which can occur when a damaged battery cell overheats and ignites. 
* This can also be used to ensure the safety and reliability of batteries used in aircrafts. Dents or deformations in battery cells can occur due to vibrations, shocks, or impacts during flight. Detecting these issues early on can help to prevent catastrophic failures and ensure that the batteries meet the strict safety standards required for aviation applications. 
* We have used ONNX to train the model which makes it framework agnostic, making it independent of any set framework. This allows developers to move between frameworks depending on what works best for the given development process.

## References
* Roboflow - [https://roboflow.com/models/object-detection](https://roboflow.com/model/yolov8)
* Ultralytics documentation - https://docs.ultralytics.com/
* YOLO - https://arxiv.org/pdf/1506.02640.pdf
* YOLO Object Detection using OpenCV and Python - https://towardsdatascience.com/yolo-object-detection-with-opencv-and-python-21e50ac599e9#:~:text=What%20is%20YOLO%20exactly%3F,in%20C%20from%20the%20author).
* ONNX - https://towardsdatascience.com/onnx-the-standard-for-interoperable-deep-learning-models-a47dfbdf9a09
* Tensorflow Lite - https://www.tensorflow.org/lite/examples/object_detection/overview
