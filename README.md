# Chess-Piece-Classification
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
<img align="left" src="https://github.com/Dion11235/Chess-Piece-Classification/blob/main/images/loss.jpeg"> 
The plot on the left hand side shows a 3d projection of the high dimensional loss surface of ResNet, 
So clearly the quality of a solution (how close to the actual global minima) depends on 
the initial starting point immensely. In neural networks those points are nothing but weights. 
Now just a zero initialization may not be practical always when we have such a small dataset. 
We need a better headstart instead. So Using Transfer Learning seems a better approach, as this gives 
a set of pretrained weights on a bigger dataset and that can be used as the initial point in our loss surface.


<img align="right" src="https://github.com/Dion11235/Chess-Piece-Classification/blob/main/images/earthquake_sgd.gif">   

Now what we will be doing hhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhkkggghdgdhnfjhd 

