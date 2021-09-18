## German Traffic Sign Detection
**Assignment work done as part of Udemy Self-Driving Car Course**

This project builds a Convolutional Neural Network that is trained on 43 different classes of traffic sign images. These kinds of classifiers built on large-scale data can be used for self-driving cars to recognize and classify traffic signs on the fly and make quick decisions - such as speed up or slow down, stop when it comes across a stop sign etc. 

The model was trained on ~35k color images of traffic signs and was test metrics were evaluated using ~12k images. In addition, you can also add a URL to an image and classify that. The output would look something like: 

![](https://github.com/sreenithibalasu/image_classifier/blob/main/images/Figure_1.png)

### Steps taken to build the model:
- Dataset Exploration - [`dataset_exploration.ipynb`](https://github.com/sreenithibalasu/image_classifier/blob/main/dataset_exploreation.ipynb) - this notebook was made before building the model, to see what the dataset has - the number of images and it's size. It also visualizes images in different classes of data. Some data preprocessing methods were tried and I decided to do the following:
  - Convert RGB image to grayscale
  -  Equalize the image histogram
  -  Normalize the image

- The model was trained using the following parameters: 
  - Epochs: 10
  - Batch Size: 50
  - Steps per epoch for training = 34,799 // 50 = 695

### Tensorboard Output
The loss and accuracy during training and validation were visualized using tensorboard
![](https://github.com/sreenithibalasu/image_classifier/blob/main/images/Screen%20Shot%202021-09-17%20at%208.10.17%20PM.png)

### Python Scripts
[`data_preprocessing,py`](https://github.com/sreenithibalasu/image_classifier/blob/main/data_preprocessing.py) - contains functions to preprocess and return preprocessed images for training
[`training_model.py`](https://github.com/sreenithibalasu/image_classifier/blob/main/training_model.py) - contains neural network and it's different layers
[`run_training.py`](https://github.com/sreenithibalasu/image_classifier/blob/main/run_training.py) - if you want to train the model on your machine, check the usage notes for this file and run it accordingly
[`test_model.py`](https://github.com/sreenithibalasu/image_classifier/blob/main/test_model.py) - tests the saved model in `checkpoints_path` on an image URL or pickled test images

### Configuration Files
['configs.json'](https://github.com/sreenithibalasu/image_classifier/blob/main/configs.json) - before running the training and testing file, some parameters in this file has to be changed:
- `checkpoints_path` - path to store the model weights and parameters during and after training. This way, while testing the model weights can be loaded and images can be tested at ease. 
- `data_path` - path where train, test and validation pickle files are stored
- `log_dir_train` - for generating tensorboard logs. Path where training logs will be stored
- `csv_mapping` - path where the CSV file with class name mappings are found

### File Structure

```
image-classifier
| configs.json
| data_preprocessing.py
| training_model.py
| run_training.py
| test_model.py
|___ checkpoints
|    |___ training_1
|
|___ german_traffic_signs
|    | train.p
|    | test.p
|    | val.p
|___ logs
|    |___ train

```
