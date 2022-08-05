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

# 問題2のレポート  Problem 2 Report

Using LSTM+Dropout model to achieve higher accuracy on the IMDb dataset with expected accuracy over 75%
 
## LSTM

<img src="https://i2.wp.com/nttuan8.com/wp-content/uploads/2019/06/lstm.png?resize=559%2C338&ssl=1" width="500" height="200">

## Dropout

<img src="https://files.ai-pool.com/a/59df0e2cc98add51893f784916195478.png" width="500" height="200">

- Setting up LSTM with parameters as the following:
 + input_size = 300
 + hidden_size = 516
 + num_layers = 2 :stacking two LSTMs together to form a stacked LSTM, with the second LSTM taking in outputs of the first LSTM and computing the final results.
 - dropout = 0.5 introduces a Dropout layer on the outputs of each LSTM layer except the last layer.
Implement Dropout to randomly zeroes some of the elements of the input tensor with probability p using samples from a Bernoulli distribution duting training. Each channel will be zeroed out independently on every forward call:
 + p = 0.5 probability of an element to be zeroed.
- Furthermore, the outputs are scaled by a factor of 1/1-p during training. This means that during evaluation the module simply computes an identity function.
- Dropout is applied to the output of LSTM
- Forward propagation is calculated with forward. When emb (x) is executed, the input x (all token columns in the mini-batch) is embedded together. Therefore, e is a tensor with text length x batch size x 300. After that, initialize the intermediate state of the LSTM. Output contains the output state of all time steps. In most cases, the final state is required for classification.
- Use "hn” to continue the sequence for backpropagation by later passing it as an argument to the LSTM.. hn[-2 ,:,:] is the end of the forward RNN while hn[-1,:,:] is the end of the backward RNN. Finally, calculate self.li(hn) and use this as the output.
## Result
### RNN model:
epoch 0 : loss 279.06442603468895

epoch 1 : loss 243.09687647223473

epoch 2 : loss 239.68640065193176

epoch 3 : loss 238.34244325757027

epoch 4 : loss 237.49841277301311

epoch 5 : loss 236.89533491432667

epoch 6 : loss 236.408755376935

epoch 7 : loss 235.98467338085175

epoch 8 : loss 235.59556458890438

epoch 9 : loss 235.2350489348173

correct: 16873

total: 25000

accuracy: 0.67492

### LSTM + Dropout model:
epoch 0 : loss 208.35595202352852

epoch 1 : loss 173.15486887469888

epoch 2 : loss 102.85342382453382

epoch 3 : loss 64.40572276711464

epoch 4 : loss 43.27356273634359

epoch 5 : loss 26.790876372368075

epoch 6 : loss 16.730014154920354

epoch 7 : loss 12.755365838645957

epoch 8 : loss 6.49576061766129

epoch 9 : loss 4.625697691124515 correct: 21039

correct: 22185

total: 25000

accuracy: 0.8874

### LSTM+Dropout model can achieve around 88% accuracy

# 問題3のレポート  Problem 3 Report
## Dataset Explanation:
IWSLT'15 English-Vietnamese Dataset
IWSLT stands for International Workshop on Spoken Language Translation
IWSLT15 (en-vi) consists of 133,317 sentence pairs of training data, 1,553 sentence pairs of development data, and 1,268 sentence pairs of test data.
## Requirements:
Input: English Text
Output: Translated Vietnamese Text
 
## Preprocessing:
Data was preprocessed as the teacher presented in the lecture, improvements are mostly made to the model and hyperparameter tuning. 
The conversion of the first token column is unnecessary because the one that has been applied to the data of distributed IWSLT15.

## Installation:
The LSTM significantly improves the encoder・decoder model based on RNN. In order to increase the BLEU rating values, we decided to incorporate the Bidirectional LSTM model to the already provided Model from Lab work 6. 
 
 
## The encoder uses the following units:
• encemb: Encoder convolution layer (256 dimensions).
Sets the number of dimensions of the word embedder to 256 dimensions. Also padding_idx allows us to specify an ID for the padding, which is vocabidx_x['<pad>'] so that a 0 vector is returned for it. 
• encrnn: Encoder LSTM calculation unit (256,516,2, dropout = 0.5, bidirectional=True)
 
## The decoder uses the following units:
• decemb: convolution layer of the decoder (516 dimensions)
• decrnn: LSTM calculation unit of the decoder (256,516,2,dropout = 0.5, bidirectional=True) 
 
* decout: Output of the decoder (516*2 the number of vocabulary of the target language)
 
## Forward function: 
LSTM bidirectional based encoder decoder calculation of forward propagation is used for the training process. x is passed in pairs with a mini-batch of input statements and a mini-batch of output statements. x[0] gives a mini-batch of input statements and x[1] gives a mini-batch of output statements. Which is set to x, y, respectively.
 
Then x is a mini-batch with the sentence length * batch size which represents the input sentence set, and y is a mini-batch with sentence length * batch size which represents the output sentence set.
 
The outputs h and c of the encoder are used as intermediate states h and c of the decoder. It saves the hidden state and the cell state, which are initially filled with zeros.
 
The squeeze function is used to manipulate tensors by removing all dimensions of an input of size 1.
The unsqueeze function is used to generate a new tensor as output by adding a new dimension of size 1 to the desired position. Again, the obtained output data and elements remain the same in the tensor.

## Execution Result:
If the learning rate was LR=0.001, epoch = 10 and the maximum sentence length at inference was 30, it could be calculated in about 70 minutes using the GPU. The BLEU score resulted from the test was 0.0923 (9.23%). However, after training for 20 epochs, The result improved to 0.0937 which was not significant.
## Conclusion:
The training didn’t result greatly in using the Bidirectional LSTM and Dropout method. A more powerful model such as Seq2Seq and Transformer would’ve produced a better BLEU score.
