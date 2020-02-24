# Social LSTM using PyTorch for Vehicle Data

This code/implementation is edited version of [Anirudh Vemula](https://github.com/vvanirudh/social-lstm-pytorch)'s code. It is edited for vehicle trajectory data . If you are using this code for your work, please cite the [original paper](http://cvgl.stanford.edu/papers/CVPR16_Social_LSTM.pdf) and Anirudh Vemula's original code.

## Getting Started

The dataset avaliable is normalized between -1 and 1. Also this version of code is only for GPU's.


## Requirements
* Python 3.6
* Seaborn (https://seaborn.pydata.org/)
* PyTorch 0.4 (http://pytorch.org/)
* Numpy
* Matplotlib
* Scipy
* GPU

## How to Run
* Before running the code, create the required directories by running the script `make_directories.sh`
* Unzip the data files inside the `data_vehicles` folder
* To train the model run `python3 social_lstm/train.py` (With default parameters)
* To test the model run `python3 social_lstm/sample.py --epoch=n` where `n` is the epoch at which you want to load the saved model. (Also since we use validation, by the end of training you should see the best epoch)
* To visualize and plot the grid run `python3 social_lstm/visualize.py` with default parameters


