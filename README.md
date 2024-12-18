# DAS Event Detection with MAEs

## Data
the data for this project is located locally on Ellie's lab's drives.

## Source Code

dataloader.py
- creates a custom dataset for the data

mae.py
- the MAE model itself

train_model.py
- will train an MAE model

finetuning.py
- finds the optimal parameters for the MAE models
- saves the model and the encoder

classifier.py
- will train a binary classifier ontop of the trained encoder from train_model.py

classifier_finetuning.py
- finds the optimal parameters for the binary classifier
