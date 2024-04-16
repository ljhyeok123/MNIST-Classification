# MNIST Classification
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1krbv80qgcDw30kySeCAfevKOIdO-Mbw-?usp=sharing)

## model.py
 Implement LeNet-5 and your custom MLP models in model.py.
 custom MLP model should have about the same number of model parameters with LeNet-5

- LeNet-5 parameters
- First Layer (conv1)
  
    Input channels: 1
  
    Output channels: 6
  
    Kernel size: 5x5

  Parameters: (Input channels * Output channels * Kernel height * Kernel width) + Output channels

  Parameters = (1 * 6 * 5 * 5) + 6 = 156

- custom MLP model

   First layer (fc1): 784 × 60 + 60 = 47 , 100 784×60+60=47,100 parameters 
   Second layer (fc2): 60 × 40 + 40 = 2 , 440 60×40+40=2,440 parameters 
   Third layer (fc3): 40 × 10 + 10 = 410 40×10+10=410 parameters 
   custom MLP model total parameters: 49,950
