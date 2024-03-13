# Part 1 Solution of Backprogation


![Example Image](photo_1.png)

Above is the image of backpropagation calculation 

![Example Image](photo_2.png)
Above is the step 1 

Below is the line by line description of step 1

1) For calculating h1,weights 1 is being multiply with input 1 + weights 2 is being multiply with input 2
2) For calculating h2,weights 3 is being multiply with input 1 + weights 4 is being multiply with input 2
3) For calculating a_h1, we will apply sigmoid function to h1
4) For calculating a_h2, we will apply sigmoid function to h2
5) Same for calculating o1 we will multiply weights 5 with a_h1 + weights 6 with a_h2
6) Same for calculating o2 we will multiply weights 7 with a_h1 + weights 8 with a_h2
7) For calculating a_o1 we will apply sigmoid function to o1
8) For calculating a_o2 we will apply sigmoid function to o2
9) For calculating E_total we will sum E1 and E2

![Example Image](photo_3.png)
Above is the step 2

In Step 2, we will calculate derivative of E_total wrt to derivative of weight 5, for calculating E_total we have to calculate derivative of E1 and E2 wrt to weight 5. but DE1( Derivative of E1 ) is dependent on W5 and DE2 is not dependent on W5

![Example Image](photo_4.png)
Above is the step 3

In Step 3, we will calculate Derivative of E_Total wrt to W5,W6,W7 and W8

![Example Image](photo_5.png)

Above is the loss graph when the learning rate is set to 0.1. When epochs goes to 66, the loss decreasing speed gets reduced

![Example Image](photo_6.png)

Above is the loss graph when the learning rate is set to 0.2. When epochs goes to 66, the loss is good as compared to 0.1 learning rate

![Example Image](photo_7.png)

Above is the loss graph when the learning rate is set to 0.3. When epochs goes to 66, the loss is even good as compared to 0.1 and 0.2 learning rate

![Example Image](photo_8.png)

Above is the loss graph when learning rate is set 0.8

![Example Image](photo_9.png)

Above is the loss graph when learning rate is set 1

![Example Image](photo_10.png)

Above is the loss graph when learning rate is set 2

# Part 2 - In part 2 we have to achieve some targets and accuracy for our models
We have implemented the below concepts which are taught in last 5 lectures
1) How many layers,
2) MaxPooling,
3) 1x1 Convolutions,
4) 3x3 Convolutions,
5) Receptive Field,
6) SoftMax,
7) Learning Rate,
8) Kernels and how do we decide the number of kernels?
9) Batch Normalization,
10) Image Normalization,
11) Position of MaxPooling,
12) Concept of Transition Layers,
13) Position of Transition Layer,
14) DropOut
15) When do we introduce DropOut, or when do we know we have some overfitting
16) The distance of MaxPooling from Prediction,
17) The distance of Batch Normalization from Prediction,
18) When do we stop convolutions and go ahead with a larger kernel or some other alternative (which we have not yet covered)
19) How do we know our network is not going well, comparatively, very early
20) Batch Size, and Effects of batch size
21) etc (you can add more if we missed it here)

## Our targets for part 2 assignment was make a model in such a way that achieves
1) 99.4% validation accuracy
2) Less than 20k Parameters
3) You can use anything from above you want. 
4) Less than 20 Epochs
5) Have used BN, Dropout,
6) (Optional): a Fully connected layer, have used GAP.

### Our model parameters is 14896


Total params: 14,896
Trainable params: 14,896
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.00
Forward/backward pass size (MB): 1.06
Params size (MB): 0.06
Estimated Total Size (MB): 1.12

and our test accuracy was 99.47% in epoch 19th

Epoch  19
loss=0.057934198528528214 batch_id=468: 100%|██████████| 469/469 [00:17<00:00, 26.84it/s] 

Test set: Average loss: 0.0172, Accuracy: 9947/10000 (99.47%)




