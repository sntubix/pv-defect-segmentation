# Multi-Class Semantic Segmentation of Photovoltaic Module Defects and Features: Towards Industrial Robotic Applications
[[Paper]()]

![IEA25_v7](https://github.com/user-attachments/assets/30341589-63ac-4de5-9eab-87ef581a2672)

**Abstract**
Automated defect detection in photovoltaic (PV) modules is essential for their maintenance and efficiency, yet challenges such as limited and imbalanced datasets hinder the adoption of high-accuracy systems. 
This study evaluates six semantic segmentation architectures based on U-Net and SegNet, paired with VGG16, MobileNet, and ResNet50 encoders, and trained on the 29-class dataset of PV module electroluminescence (EL) images. 
To address dataset imbalance, custom class weights were applied for all the feature and defect classes. VGG16-UNet outperformed other architectures, achieving a mean intersection over union(IoU) of 0.663 for feature classes and 0.326 across defect classes. 
In particular, it improved the detection of rare defects, such as dead cell, by 0.129 IoU. 
While previous research focused on a specific subset of classes, this study is the first to provide a comprehensive performance evaluation across all classes.
It establishes a baseline for multi-class semantic segmentation in PV defect detection, laying the groundwork for further industrial applications such as in-field defect detection integrated into solar panel cleaning robots.

## Citation 
