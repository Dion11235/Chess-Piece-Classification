# Chess-Piece-Classification with different Optimizers
AI in Sports is not a new thing. People have been interested always to study player statistics and worth, 
and for that analysis we need to record how good the player is performing. In Chess the record of the moves played so far helps the analysis of the whole game.

Now, registering the moves manually, espescially in a rapid chess match, is a tideous job. This project automates the procedure of detecting chess pieces.
detecting the pieces over successive board positions can give us exactly what we are looking for.

## Dataset
the dataset that has been used here can be found in this link : https://www.kaggle.com/niteshfre/chessman-image-dataset  
A quick look into the dataset:  

**Training data** -
![Training Data](https://github.com/Dion11235/Chess-Piece-Classification/blob/main/plots/train_data.png)  

**Validation data** -
![Validation data](https://github.com/Dion11235/Chess-Piece-Classification/blob/main/plots/valid_data.png)

Clearly the data contains not only real pieces but also some cartoon figures, 
that would help the model to generalize and consider the small features specific to the pieces. But the dataset is small,  
>training data sizes :  ['Bishop' : 66, 'Queen' : 60, 'Pawn' : 84, 'Rook' : 80, 'King' : 60, 'Knight' : 82]  
>validation data sizes :  ['Bishop' : 21, 'Queen' : 19, 'Pawn' : 26, 'Rook' : 25, 'King' : 19, 'Knight' : 26] 

To deal with the less data issue, Data Augmentations have been used.

## Model
<p align="center">
  <img width="300" height="300" src="https://github.com/Dion11235/Chess-Piece-Classification/blob/main/images/loss.jpeg"><br>
  <b>Source :</b><br>
  <a href="https://arxiv.org/pdf/1712.09913.pdf">Visualizing Loss Surface of Neural Networks</a>
</p>



The plot on the left hand side shows a 3d projection of the high dimensional loss surface of ResNet, 
So clearly the quality of a solution (how close to the actual global minima) depends on 
the initial starting point immensely. In neural networks those points are nothing but weights. 
Now just a zero initialization may not be practical always when we have such a small dataset. 
We need a better headstart instead. So Using Transfer Learning seems a better approach, as this gives 
a set of pretrained weights on a bigger dataset and that can be used as the initial point in our loss surface. I have used vgg16 encoder pretrained on Imagenet dataset, and then I have trained the classifier layers over our Chessman dataset.

<p align="center">
  <img width="500" height="500" src="https://github.com/Dion11235/Chess-Piece-Classification/blob/main/images/earthquake_sgd.gif"><br>
  <b>Source :</b><br>
  <a href="http://neupy.com/2019/06/10/earthquakes_in_neural_network_landscape.html">Earthquake in Landscape of Neural Networks</a>
</p>

Now training neural networks with taking one batch of data every time does not distort the actual gradient, but changes the whole loss surface itself. 
So A bigger batch size leads to less "earthquake" in loss surface hence closer to the actual minima. I have used batch size of 30 here, one can try a bigger batch, but that would require greater computation power. Now, lets look into the performances of the optimizers - SGD, Adagrad, RMSprop, Adam, with learning rate 0.001.

## Comparative study of the Optimizers :
### SGD :
<p align="center">
  <img src="https://github.com/Dion11235/Chess-Piece-Classification/blob/main/plots/SGD/lr=0.001/loss_acc.png"><br>
  <img src="https://github.com/Dion11235/Chess-Piece-Classification/blob/main/plots/SGD/lr=0.001/grad_norms.png">
</p>

The second image shows the distribution of the gradients of The three Classifier layers in the fully connected network. An interesting fact to observe is how the gradients get more drifted away from zero as we go towards the terminal layers. Here Layer-3 is the last layer of the classifier section. The most logical explanation of this phenomenon , I can guess, is that may be we are losing information more and more as we go deeper.

**sample prediction** :  
<p align="left">
  <img src="https://github.com/Dion11235/Chess-Piece-Classification/blob/main/plots/sgd_pred.png"><br>
</p>

**class accuracies** :  
`Accuracy for class Bishop :  0.7619047619047619`  
`Accuracy for class Queen :  0.7894736842105263`  
`Accuracy for class Pawn :  0.7307692307692307`  
`Accuracy for class Rook :  0.84`  
`Accuracy for class King :  0.7894736842105263`  
`Accuracy for class Knight :  0.9615384615384616`

we can see the model is pretty confident about the knight (because of its distinguishable figure), but not so with the others. 

### Adagrad :
Adaptive Gradient Algorithm (Adagrad) is an algorithm for gradient-based optimization. The learning rate is adapted component-wise to the parameters by incorporating knowledge of past observations. It performs larger updates (e.g. high learning rates) for those parameters that are related to infrequent features and smaller updates (i.e. low learning rates) for frequent one. It performs smaller updates As a result, it is well-suited when dealing with sparse data (NLP or image recognition) Each parameter has its own learning rate that improves performance on problems with sparse gradients.
<p align="center">
  <img src="https://github.com/Dion11235/Chess-Piece-Classification/blob/main/images/adagrad.png"><br>
</p>

<p align="center">
  <img src="https://github.com/Dion11235/Chess-Piece-Classification/blob/main/plots/Adagrad/lr=0.001/loss_acc.png"><br>
  <img src="https://github.com/Dion11235/Chess-Piece-Classification/blob/main/plots/Adagrad/lr=0.001/grad_norms.png">
</p>

The plots show that Adagrad took less time and gave more accuracy than SGD. Also The grads seem to be more close to zero in this case. Lets look at the class based accuracies.

**Sample Prediction** :
<p align="left">
  <img src="https://github.com/Dion11235/Chess-Piece-Classification/blob/main/plots/adagrad_pred.png"><br>
</p>

**class accuracies** :  
`Accuracy for class Bishop :  0.7142857142857143`  
`Accuracy for class Queen :  0.8421052631578947`  
`Accuracy for class Pawn :  0.7692307692307693`  
`Accuracy for class Rook :  0.92`  
`Accuracy for class King :  0.9473684210526315`  
`Accuracy for class Knight :  1.0`

More or less every class is better classified with Adagrad, lets see if we can improve the performance further.

### RMSprop :
RMSprop is a gradient-based optimization technique used in training neural networks. It was proposed by the father of back-propagation, Geoffrey Hinton. Gradients of very complex functions like neural networks have a tendency to either vanish or explode as the data propagates through the function (refer to vanishing gradients problem). Rmsprop was developed as a stochastic technique for mini-batch learning.

RMSprop deals with the above issue by using a moving average of squared gradients to normalize the gradient. This normalization balances the step size (momentum), decreasing the step for large gradients to avoid exploding and increasing the step for small gradients to avoid vanishing. Simply put, RMSprop uses an adaptive learning rate instead of treating the learning rate as a hyperparameter. This means that the learning rate changes over time.

<p align="center">
  <img width="300" height="300" src="https://github.com/Dion11235/Chess-Piece-Classification/blob/main/images/rmsprop.png"><br>
</p>

<p align="center">
  <img src="https://github.com/Dion11235/Chess-Piece-Classification/blob/main/plots/RMSprop/lr=0.001/lr=0.001loss_acc.png"><br>
  <img src="https://github.com/Dion11235/Chess-Piece-Classification/blob/main/plots/RMSprop/lr=0.001/lr=0.001grad_norms.png">
</p>

**sample prediction** :
<p align="left">
  <img src="https://github.com/Dion11235/Chess-Piece-Classification/blob/main/plots/rmsprop_pred.png"><br>
</p>

**class accuracies** :  
`Accuracy for class Bishop :  0.8571428571428571`  
`Accuracy for class Queen :  1.0`  
`Accuracy for class Pawn :  0.9230769230769231`  
`Accuracy for class Rook :  1.0`  
`Accuracy for class King :  0.9473684210526315`  
`Accuracy for class Knight :  1.0`

The high accuracies might be eplained by the fact that RMSprop deals with vanishing gradient issue. In this case, as very small features may be very important to distinguish between two pieces, for example, only the cut on the head makes the bishop different from a pawn in some cases, the small crown on the top distinguishes the queen from a rook as there is no height difference in image.

### Adam :
Adam can be looked at as a combination of RMSprop and Stochastic Gradient Descent with momentum. It uses the squared gradients to scale the learning rate like RMSprop and it takes advantage of momentum by using moving average of the gradient instead of gradient itself like SGD with momentum.

<p align="center">
  <img width="300" height="100" src="https://github.com/Dion11235/Chess-Piece-Classification/blob/main/images/adam_eqn.png"><br>
  <img width="200" height="200" src="https://github.com/Dion11235/Chess-Piece-Classification/blob/main/images/adam_eqn2.png"><br>
  <img width="300" height="100" src="https://github.com/Dion11235/Chess-Piece-Classification/blob/main/images/adam_eqn3.png"><br>
</p>

<p align="center">
  <img src="https://github.com/Dion11235/Chess-Piece-Classification/blob/main/plots/Adam/lr=0.001/lr=0.001loss_acc.png"><br>
  <img src="https://github.com/Dion11235/Chess-Piece-Classification/blob/main/plots/Adam/lr=0.001/lr=0.001grad_norms.png">
</p>

**sample prediction** :
<p align="left">
  <img src="https://github.com/Dion11235/Chess-Piece-Classification/blob/main/plots/Adam_pred.png"><br>
</p>

**class accuracies** :  
`Accuracy for class Bishop :  0.7619047619047619`  
`Accuracy for class Queen :  0.8947368421052632`  
`Accuracy for class Pawn :  0.9230769230769231`  
`Accuracy for class Rook :  0.92`  
`Accuracy for class King :  0.8947368421052632`  
`Accuracy for class Knight :  1.0`

### A comparison table of dfferent optimizers over different classes :
<p align="left">
  <img width="800" height="400" src="https://github.com/Dion11235/Chess-Piece-Classification/blob/main/plots/comparison.png"><br>
</p>

Thank you.
