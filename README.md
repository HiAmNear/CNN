# CNN
CNN Project - AI Learning

# 問題1のレポート  Problem 1 Report
## Dataset Explanation:
The CIFAR-10 dataset consists of 60000 32x32 colour images in 10 classes, with 6000 images per class. There are 50000 training images and 10000 test images.
The dataset is divided into five training batches and one test batch, each with 10000 images. The test batch contains exactly 1000 randomly-selected images from each class. The training batches contain the remaining images in random order, but some training batches may contain more images from one class than another. Between them, the training batches contain exactly 5000 images from each class.

<img src="https://i0.wp.com/cvml-expertguide.net/wp-content/uploads/2022/03/CIFAR-10.png?fit=484%2C378&ssl=1" width="500" height="200">
 
Classes name dataset include : airplane,automobile,bird,cat,deer,dog,frog,horse,ship,truck
## Requirements:
·	Input : 32x32 colour images in 10 classes

·	Ouput : images concern 1 class in 10 classes of input
## Model:
There are various types of CNN, but the CNN called VGG16 is famous for its relatively simple structure and high performance. VGG16 is a CNN from 13 convolution layers and 3 FC layers, and has the configuration shown in the picture below.

<img src="https://neurohive.io/wp-content/uploads/2018/11/vgg16-1-e1542731207177.png" width="500" height="200">

224x224x3 RGB image data is used as input. First, there are two convolution layers + ReLU, which are converted to 224x224x64 data. Notice that the three-color features have been converted to 64-dimensional features.
Then use maximum pooling to halve the image size vertically and horizontally (112x112x64). Next, the convolution layer + ReLU is passed through two layers and converted to 112x112x128 data. 

Then, similarly, convert to 56x56x128 with maximum pooling, and then convert to 56x56x256 data with 3 convolution layers. Then, it is converted to 28x28x512 data in the same way, and it is also converted to 14x14x512 data in the same way. After this, maximum pooling is applied to convert to 7x7x512. In other words, the entire image is now divided into 7x7 regions, and 512-dimensional vectors are calculated in each region. 

After that, it is converted to a 1x1x4096 vector using 2 layers of FC layer + ReLU, converted to a 1x1x1000 dimensional vector for classification (1000 class classification) using 1 layer FC, and then soft. Applying Softmax, The data size of each layer is arranged in an easy-to-understand manner as follows.

Conv → BatchNorm2d → ReLU (→ Pooling) is a typical element.
    FC → ReLU when it gets closer to the output layer.
    The output layer is FC → Softmax.
    After initializing the model, add it
    → 64 channel 2x convolution layer with the same padding as the 3x3 kernel
    → 2x2 pool size and stride 2x2 1xmax pool layer
    → 128-channel 2x convolution layer with the same padding as the 3x3 kernel
    → 2x2 pool size and stride 2x2 1xmax pool layer
    → 256-channel 3x convolution layer with the same padding as the 3x3 kernel
    → 2x2 pool size and stride 2x2 1xmax pool layer
    → 512 channel 3x convolution layer with the same padding as the 3x3 kernel
    → 2x2 pool size and stride 2x2 1xmax pool layer
    → 512 channel 3x convolution layer with the same padding as the 3x3 kernel
    → 2x2 pool size and stride 2x2 1xmax pool layer
 
BatchNorm2d is always added after the convolutional layer of the convolutional neural network, data normalization. This prevents the data from becoming too large and destabilizing the network performance before ReLU, and is often used in convolutional networks (lost or exploding). Also, add a ReLU activation to each layer so that all negative values are not passed to the next layer.
 
In the forward method, after receiving the input x and applying block1_output, block2_output is applied, then block3_output is applied, and block4_output is applied again. Finally, apply block5_output, x = self.avgpool (x), and make the output x as it is.

## Result
The result can be calculated in 15 minutes, using LR = 0.0001, 10 epoch, and GPU. The accuracy was 0.8032.

