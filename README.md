# **Brain Tumor Classification with CNN**

In this project, I used a convolutional neural network to classify brain MRI's into four catagories: No Tumor, Glioma, Meningioma and Pituary. 

The network was built using TensorFlow/ Keras, and my best model achieved 93.04% accuracy.

## Dataset

Training was done using [this](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset) dataset of 1311 brain MRIs. Each category (No Tumor, Glioma, Meningioma and Pituary) has roughly the same number of images, so class imbalance was not a big issue. 
Due to the small size of the dataset, I chose not to set any images aside for validation. This made it more difficult to tweak the architecture and hyperparameters, but after some trial and error I was able to achieve good results.

## Architecture and Training

* Input and Rescaling Layer (224,224,3)
* Convolutional Layer 1: 32 Filters, 3x3 Kernal Size, Relu Activation
* Convolutional Layer 2: 64 Filters, 3x3 Kernal Size, Relu Activation
* Convolutional Layer 3: 128 Filters, 3x3 Kernal Size, Relu Activation
* Convolutional Layer 4: 128 Filters, 3x3 Kernal Size, Relu Activation
* Global Average Pooling
* Dense Layer 1: 256 Neurons, Relu Activation
* Dropout Layer 1: 0.3 Dropout Rate
* Dense Layer 2: 128 Neurons, Relu Activation
* Dropout Layer 2: 0.5 Dropout Rate
* Dense Layer 3: 4 Neurons, Softmax Activation

Batch-Normalization and Max-Pooling is performed after every convolutional layer.

Batch Size: 32

Epochs: 25

Learning Rate: 0.001

Optimizer: Adam

## Accuracy

**Overall:** 93.04%

**Meningioma:** 84.6%

**No Tumor:** 98%

**Glioma:** 86.3%



