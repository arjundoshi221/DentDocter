# Cell dent detection

## Introduction

## Data

## Model

As a solution to our object detection problem, we chose the yolo-v8 architecture. The v8 architecture was made public and is maintained by [Ultralytics](https://ultralytics.com/).

YOLO or you only look once, is a real-time object detection algorithm. It was introduced in 2016 in the research paper [You Only Look Once: Unified, Real-Time Object Detection](https://arxiv.org/pdf/1506.02640v5.pdf). Compared to other regional proposal classification networks (Fast RCNN and Faster RCNN), which perform detection on various region proposals and thus end up performing prediction multiple times for various regions in a image, YOLO architecture is more like a FCNN (fully convolutional neural network) and passes the image ($n * n$) once through the FCNN and the output is ($m * m$) prediction.

Why YOLO is preferred over other object detection alogrithm:

- **Speed**: This algorithm improves the speed of detection because it can predict objecst in real-time
- **High Accuracy**: YOLO is a predictive technique that provides accurate results with minimal background errors.
- **Learning capabilities**: The algorithm has excellent learning capabilities that enable it to learn the representations of objects and apply them in object detection.

### How YOLO works

The YOLO algorithm works using the following three techiques:

- **Residual block**: First, the image is divided into various grids. Each grid has a dimension of $s*s$.
  ![alt.png](https://www.guidetomlandai.com/assets/img/computer_vision/grid.png)

  In the above image, there are many grid cells of equal dimension. Every grid cell will detect objects that appear within them. For example, if an object center appears within a grid cell, then this cell will be responsible for detecting it.

- **Bounding box regression**: A bounding box is an outline that highlights an object in an image. Every bounding box in the image consists of the following attributes:
  - Width (bw)
  - Height (bh)
  - Class (for example: person, car, traffic light)
  - Boudning box center (bx, by)
    ![bb.png](https://www.section.io/engineering-education/introduction-to-yolo-algorithm-for-object-detection/bounding-box.png)
    YOLO uses a single bounding box regression to predict the height, width, center and class of objects.
- **IOU** or Intersection Over Union: This is a phenomenon in object detection that describes how boxes overlap. YOLO uses IOU to provide an output box that surrounds the objects perfectly.
  Each grid cell is responsible for predicting the bounding boxes and confidence scores. The IOU is equal to 1 if the predicted bounding box is the same as the real box. This mechanism eliminates bounding boxes that are not equal to the real box.

## Usage

## References
