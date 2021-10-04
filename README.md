# MobileNet-1D-2D-Tensorflow-Keras  
Supported Models: 
1. MobileNetV1 [1]
2. MobileNetV2 [2]
3. MobileNetV3 (Small, Large) [3]

## MobileNetV1  
The MobileNetV1 Architeture implements a Standard Convolution Layer and a DepthWise Separable Convolution Layer iteratively in turn. The DepthWise Separable Convolution Layers consist of a series of a DepthWise and a PointWise Convolution Layer. The constituting blocks for both types of Convolutional Layers are shown below [1]:  
![Standard vs. DepthWise Conv Blocks](https://github.com/Sakib1263/MobileNet-1D-2D-Tensorflow-Keras/blob/main/Documents/Images/DW_Blocks.png "Standard vs. DepthWise Conv Blocks")  
MobileNetv1 also uses a Stride Size (s) of 1 and 2 by turn. Model Width or Filter Number is doubled after eeach pair of Conv Layers. MobileNetV1 implemented BatchNormalization in their convolutional layers. The MobileNetV1 Body Architeture is shown in detail in the table below [1]:  
![MobileNetV1 Architeture](https://github.com/Sakib1263/MobileNet-1D-2D-Tensorflow-Keras/blob/main/Documents/Images/MobileNetv1_Architecture.png "MobileNetV1 Architeture")  

## MobileNetV2  
MobileNetV2 uses Inverted Residual Blocks instead of traditional Residual Blocks in their BottleNeck Layers. Both types of blocks are depicted below in visual comparison. More in this paper [2].  
![Inverted vs. Common Residual Blocks](https://github.com/Sakib1263/MobileNet-1D-2D-Tensorflow-Keras/blob/main/Documents/Images/Inverted_Residual.png "Inverted vs. Common Residual Blocks")  

The BottleNeck Layers are two types based on the Stride Length (1 or 2), as shown in the figure below. For Stride = 1, MobileNetV2 
Mentionable that for each set of blocks with stride = 2, only the first BottleNeck block has a stride of 2 while rest has a stride of 1. On the contrary, for set of blocks with stried = 1, BottleNeck Layers have a stride of 1. The bottleneck blocks of MobileNetV2 differs from that of MobileNetV1 and other similar Deep Networks such as ShuffleNet or NasNet.  
![BottleNeck Layer MobileNetv2](https://github.com/Sakib1263/MobileNet-1D-2D-Tensorflow-Keras/blob/main/Documents/Images/MobileNetv2vsOthers.png "BottleNeck Layer MobileNetv2")  
The MobileNetV2 architecture consists of a sandwitch of multiple Bottleneck Blocks between two convolutional blocks as shown in the model architecture below. Filter size and other parameters are change in each set of blocks.  
![MobileNetV2 Architecture](https://github.com/Sakib1263/MobileNet-1D-2D-Tensorflow-Keras/blob/main/Documents/Images/MobileNetv2.png "MobileNetV2 Architecture")  

## MobileNetV3 (Small, Large)
The MobileNetV3 architecture brings in the concept of "Squeeze and Excite" [3] in place of MobileNetV2's BottleNeck Residual Blocks, as shown below:  
![Squeeze and Excite](https://github.com/Sakib1263/MobileNet-1D-2D-Tensorflow-Keras/blob/main/Documents/Images/MNv3_Block.png "Squeeze and Excite")  
MobileNetV3 also proposes an efficient last layer [3] (Densely Connected Layer) compared to the one in MobileNetV2 and V1.  
![MV3 Efficient Last Layer](https://github.com/Sakib1263/MobileNet-1D-2D-Tensorflow-Keras/blob/main/Documents/Images/MV3_Last_Layer.png "MV3 Efficient Last Layer")  
MobileNetV3 paper [3] proposes a lighter and a denser version of the same architecture, namely MobileNetV3_Small and MobileNetV3_large, respectively. Both architectures are shown below:  

**MobileNetV3_Small**  
![MobileNetV3_Small Architecture](https://github.com/Sakib1263/MobileNet-1D-2D-Tensorflow-Keras/blob/main/Documents/Images/MBNet_v3_Small.png "MobileNetV3_Small Architecture")  
**MobileNetV3_Large**  
![MobileNetV3_Large Architecture](https://github.com/Sakib1263/MobileNet-1D-2D-Tensorflow-Keras/blob/main/Documents/Images/MBNet_v3_Large.png "MobileNetV3_Large Architecture") 

## Supported Features  
The speciality about this model is its flexibility. The user has the option for: 
1. Choosing any of 4 available MobileNet models of 3 versions for either 1D or 2D tasks.
2. Varying number of input kernel/filter, commonly known as the Width of the model.
3. Varying number of classes for Classification tasks and number of extracted features for Regression tasks.
4. Varying number of Channels in the Input Dataset.  
5. Controlling the 'alpha' parameter to control the model size, default is 1.0.  
6. Option for Dropout in the MLP block, just before the final output.

## References  
**[1]** Howard, A., Zhu, M., Chen, B., Kalenichenko, D., Wang, W., & Weyand, T. et al. (2021). MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications. arXiv.org. Retrieved 31 August 2021, from https://arxiv.org/abs/1704.04861.  
**[2]** Sandler, M., Howard, A., Zhu, M., Zhmoginov, A., & Chen, L. (2021). MobileNetV2: Inverted Residuals and Linear Bottlenecks. arXiv.org. Retrieved 31 August 2021, from https://arxiv.org/abs/1801.04381.  
**[3]** Howard, A., Sandler, M., Chu, G., Chen, L., Chen, B., & Tan, M. et al. (2021). Searching for MobileNetV3. arXiv.org. Retrieved 31 August 2021, from https://arxiv.org/abs/1905.02244.  
