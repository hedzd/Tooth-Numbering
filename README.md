# Tooth-Numbering

### Project Description
The goal of this project was numbering tooth on panoramic dental X-rays using object detection models. In order to obtain a satisfactory result, two main models were trained:
 - **Quadrant detection model**
 - **Enumeration model**

The quadrant detection model segment a panoramic X-ray  into 4 quadratns(upper left, upper right, lower left, and lower right). After that, each cropped quadrant is directly fed into the enumeration model and the result can be merged afterwards.

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
A private dataset collected and labeled by researchers at SBMU was used. The dataset consists of 1007 panoramic X-ray images with annotation.
