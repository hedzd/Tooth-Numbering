# Tooth-Numbering

### Project Description
The goal of this project was numbering tooths on panoramic dental X-rays using object detection models. In order to obtain a satisfactory result, two main models were trained:
 - **Quadrant detection model:** A panoramic X-ray is fed into the quadrant detection model to be segmented into 4 quadrants(upper left, upper right, lower left, and lower right). Then, each detected quadrant is cropped with a margin. After that, left side images and bounding boxes were flipped. 
 - **Enumeration model**:
Each cropped quadrant is directly fed into the enumeration model and the result can be merged afterwards.


### Implementation and results
Models were implemented using Detectron2 library. Here are some results from **Faster R-CNN** models:  
- Quadrant detection:  
**Evaluation results:** AP: 68.46, AP50: 84.37  
**Predicted boxes visualisation:** 

![quadrants](https://i.ibb.co/swmMbN7/download-5.png)
- Tooth detection and numbering:  
**Evaluation results:** AP: 72.74, AP50: 95.05  
**Predicted boxes visualisation:** 
  
![quadrants](https://i.ibb.co/cJ90MTT/download.png)
### Dataset
A private dataset collected and labeled by SBMU researchers was used. The dataset consists of 1007 panoramic X-ray images with annotation.
