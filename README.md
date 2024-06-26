# MNIST Classification

Designed based on the Artificial Neural Networks and Deep Learning course(2024-1) assignment template from Seoul National University of Science and Technology.

The design was done in the Google Colab environment and can be seen in the notebook below. 

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1krbv80qgcDw30kySeCAfevKOIdO-Mbw-?usp=sharing)

## dataset.py
1) Each image was preprocessed as follows: 
    - All values in the range [0,1]. 
    - Divided by a sub-mean of 0.1307, standard deviation of 0.3081
   - This preprocessing is implemented using torchvision.transforms
2) Get the following labels from the file name: {number}_{label}.png


## model.py

The number of model parameters in the custom MLP model was set to approximately the same number as in LeNet-5. LeNet-5 has a total of 44526 parameters and the custom MLP has a similar number, 49950 parameters.


- How to calculate parameters?
  
  Convolutional layer Parameters: (Input channels * Output channels * Kernel height * Kernel width) + Output channels


  Fully connected layer Parameters: (Input neurons * Output neurons) + Output neurons

  
- LeNet-5 parameters
  - First Layer (conv1)
  
    Input channels: 1
  
    Output channels: 6
  
    Kernel size: 5x5

    Parameters = (1 * 6 * 5 * 5) + 6 = 156
  - Second Layer (conv2)
  
    Input channels: 1
  
    Output channels: 6
  
    Kernel size: 5x5

    Parameters = (1 * 6 * 5 * 5) + 6 = 156

   - Third Layer (fc1)
 
     Input neurons: 16 * 4 * 4

     Output neurons: 120

     Parameters = (16 * 4 * 4 * 120) + 120 = 30840

    - Fourth Layer (fc2)

      Input neurons: 120
      
      Output neurons: 84

      Parameters = (120 * 84) + 84 = 10164

     - Fifth Layer (fc3)

       Input neurons: 84
       
       Output neurons: 10
       
       Parameters = (84 * 10) + 10 = 850

     - Total parameters: 44526

- custom MLP model

   - First layer (fc1)

     Input neurons: 784
      
      Output neurons: 60
     
     Parameters: 784 × 60 + 60 = 47100

  - Second layer (fc2)

    Input neurons: 60
      
      Output neurons: 40
    
    Parameters: 60 × 40 + 40 = 2440

  - Third layer (fc3)

    Input neurons: 40
      
      Output neurons: 10

    Parameters: 40 × 10 + 10 = 410

   - Total parameters: 49950

## main.py

- main function are instantiated as follows

  
 1) Dataset objects for training and test datasets 
 2) Dataloaders for training and testing 
 3) A model 
 4) Optimizer: SGD with an initial learning rate of 0.01 and momentum of 0.9 5) Cost function: use torch.nn.CrossEntropyLoss

    
 In model.py, we monitor the training of LeNet-5 and our custom MLP model for 10 epochs, showing the average loss value and accuracy at the end of each epoch. For each model, you can see four plots in the figure below: loss and accuracy curves for the training and test datasets, respectively.

 The typical accuracy of LeNet-5 on the MNIST dataset is 98%. When implementing LeNet-5 and custom MLP models in model.py, the 10 epochs test accuracies are 99.01% and 97.45%, respectively. LeNet-5 has a higher test accuracy.

![image](https://github.com/ljhyeok123/MNIST-Classification/assets/146068357/6f6cb968-374d-4d21-95b4-f5634b194db9)
![image](https://github.com/ljhyeok123/MNIST-Classification/assets/146068357/ec02e952-df82-4bdb-947d-70170de94ca4)

To improve the LeNet-5 model, two normalization techniques were applied: batch normalization and dropout. We used torch.nn.BatchNorm2d to add batch normalization and torch.nn.Dropout to add dropout.  While these additional techniques can stabilize model training and reduce overfitting, many epochs will result in overfitting. The custom MLP did not decrease test loss at the 13th epoch, and LeNet-5 similarly increased test loss at the 14th epoch, resulting in little increase in test accuracy.

![image](https://github.com/ljhyeok123/MNIST-Classification/assets/146068357/5bf40e0a-cf9b-4382-b035-b3c7684cc359)
![image](https://github.com/ljhyeok123/MNIST-Classification/assets/146068357/0def0cc4-0695-4c10-9222-8acf121373a1)



 

