# 🚀 Convolutional Neural Network (CNN)

A **Convolutional Neural Network (CNN)** model built using TensorFlow/Keras.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg?style=flat-square&logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?style=flat-square&logo=tensorflow)

---

## 🏗 Model Architecture

```python
Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 conv2d (Conv2D)             (None, 26, 26, 32)        320       
 max_pooling2d (MaxPooling2D) (None, 13, 13, 32)       0         
 conv2d_1 (Conv2D)           (None, 11, 11, 64)       18,496    
 max_pooling2d_1 (MaxPooling2D) (None, 5, 5, 64)       0         
 flatten (Flatten)           (None, 1600)             0         
 dense (Dense)               (None, 128)              204,928   
 dropout (Dropout)           (None, 128)              0         
 dense_1 (Dense)             (None, 10)               1,290     
=================================================================
Total params: 225,036 (879.05 KB)
Trainable params: 225,034 (879.04 KB)
Non-trainable params: 0 (0.00 B)
Optimizer params: 2 (12.00 B)
