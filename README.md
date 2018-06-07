# CIFAR-10 CNN with PyTorch

## Accuracy / Epoch
86.02% accuracy with my own model

110 epoch

## What is CIFAR-10 Dataset?
The CIFAR-10 dataset consists of 60000 32x32 colour images in 10 classes, with 6000 images per class. There are 50000 training images and 10000 test images. 

If you want to download or learn more detail about of CIFAR-10 dataset, you can follow this [link](http://github.com).

## CNN Architecture
The architecture used in this project is shown in the figure.
![CNN Architecture](/images/cnn_new.png)

I use ELU instead of ReLU function. I searched some academic papers and internet sites. Researchs reveal that the function tend to converge cost to zero faster and produce more accurate results. Different to other activation functions, ELU has a extra alpha constant which should be positive number. 

ELU is very similiar to RELU except negative inputs. They are both in identity function form for non negative inputs. On the other hand, ELU becomes smooth slowly until its output equal to -α whereas RELU sharply smoothes. Notice that α is equal to +1 in the following illustration. 

So, ELU is a strong alternative to ReLU as an activation function in neural networks. The both functions are common in convolutional neural networks. Unlike to ReLU, ELU can produce negative outputs.

[More Detail](https://sefiks.com/2018/01/02/elu-as-a-neural-networks-activation-function/)

## Data Loading
PyTorch has a function for loading CIFAR-10 database. I used it. This function gets some parameters. Most important parameter is **transform**. I used below code to this parameter.

```python
transformtrain = T.Compose([
  T.RandomCrop (32, padding =4) ,
  T.RandomHorizontalFlip () ,
  T.ToTensor () ,
  T.Normalize ((0.4914,0.4822,0.4465),(0.2023,0.1994,0.2010)) ,
])
```


RandomCrop and RandomHorizontalFlip functions are data augmentation tools in PyTorch. I used these functions because CIFAR-10 dataset is small. RandomCropping duplicated images, RandomHorizontalFlip rotated images with 0.5 possibility. I did not use these functions for test set, just used for training set.

Last important parameter of data loading is normalizing values. I used T.Normalize for this step. Firstly, I used (0.5, 0.5, 0.5), (0.5, 0.5, 0.5) values in this function but then I read some documents and code s about of CIFAR-10, I found these values (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010). These values are calculated mean and std values for CIFAR-10 dataset.

## Training
Firstly, I used some values from config.py. (i.e. learning decay). I searched some academic papers, they are offering CrossEntropyFunction() for loss calculation. Because, CrossEntropyFunction combine softmax loss and negative log likelihood loss so it is useful when training a classification problem. 

I used Stochastic Gradient Descent because this optimization algorithm is very good for reducing cost function. Stochastic Gradient Descent minimize loss. I used other optimization algorithms like Adam and AdaGrad. I found best results in SGD. I define weight decay for L2 penalty. SGD is using L2 regularization and this is an random variable.

Updating learning rate in each epoch. If current epoch is greater than previous loss, reducing learning rate.

```python
if loss_meter.value()[0]>previousloss:
  lr = lr ∗ opt.lr_decay
  for param in optimizer.param_groups:
    param[’lr’]=lr
previousloss = loss_ meter.value()[0]
```

### Variation of Dynamic Learning Rate

![Variation of dynamic learning rate](/images/learning_rate_graph.png)

### Accuracy Graph

![Variation of dynamic learning rate](/images/accuracy_graph.png)

Lastly, I save model of state dictionary.

## Testing

I explained this values in data loading section. (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)

```python
transformtest = T.Compose([
  T.ToTensor(),
  T.Normalize((0.4914,0.4822,0.4465),(0.2023,0.1994,0.2010))
])
```

