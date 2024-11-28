## Data
data folder holds all the data we will use
- to get the data we need, download the zip from the drive
- unzip it and make sure all of the files are in a folder called Toy_dataset

## Source Code

dataloader.py
- what this really does is create the dataset from the toy dataset (this will need to be renamed to dataset) 
- we can make the dataset, with all the postional encodings, paddings, and transformations then put it into a Dataloader
- easy way to make test, val, and train sets


mae.py
- the MAE model itself

train_model.py
- will call the other two files and train an MAE with the dataset
- current error about batch size
