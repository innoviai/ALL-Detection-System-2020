# Peter Moss Acute Myeloid & Lymphoblastic Leukemia AI Research Project

## Acute Lymphoblastic Leukemia Detection System 2020

&nbsp;

# AllDS2020 Classifier

## Introduction

This project is the classifier that is used in Acute the Lymphoblastic Leukemia Detection System 2020. The network provided in this project was originally created in my [ALL research papers evaluation project](https://github.com/leukemiaresearchassociation/ALL-IDB-Classifiers/blob/master/Python/Tensorflow/2-0/Classification/ALL-Papers/Evaluations/Paper-1.md "ALL research papers evaluation project"), where I replicated the network proposed in the [Acute Leukemia Classification Using Convolution Neural Network In Clinical Decision Support System](https://airccj.org/CSCP/vol7/csit77505.pdf "Acute Leukemia Classification Using Convolution Neural Network In Clinical Decision Support System") paper by Thanh.TTP, Giao N. Pham, Jin-Hyeok Park, Kwang-Seok Moon, Suk-Hwan Lee, and Ki-Ryong Kwon, and the data augmentation proposed in  [Leukemia Blood Cell Image Classification Using Convolutional Neural Network](http://www.ijcte.org/vol10/1198-H0012.pdf "Leukemia Blood Cell Image Classification Using Convolutional Neural Network") by T. T. P. Thanh, Caleb Vununu, Sukhrob Atoev, Suk-Hwan Lee, and Ki-Ryong Kwon. The original project was inspired by the [work](https://github.com/AmlResearchProject/AML-ALL-Classifiers/blob/master/Python/_Keras/QuantisedCode/QuantisedCode.ipynb "work") done by [Amita Kapoor](https://www.leukemiaresearchassociation.ai/team/amita-kapoor/profile "Amita Kapoor") and [Taru Jain](https://www.leukemiaresearchassociation.ai/student-program/student/taru-jain "Taru Jain") and my previous [projects](https://github.com/AMLResearchProject/AML-ALL-Classifiers/tree/master/Python/_Keras/AllCNN "projects") based on their work. 

&nbsp;

## Results

| Training | Validation | Test | Accuracy | Recall | Precision | AUC/ROC |
| -------------------- | ----- | ---------- | ---------- | ---------- | ---------- | ---------- |
| 2163 |  741 | 20 | 0.97840756 | 0.97840756 | 0.97840756 | 0.995607 |

&nbsp;

# Network Architecture

<img src="https://www.leukemiaresearchfoundation.ai/github/media/images/paper_1_architecture.png" alt="Proposed Architecture" />

_Fig 1. Proposed architecture ([Source](https://airccj.org/CSCP/vol7/csit77505.pdf "Source"))_

In [Acute Leukemia Classification Using Convolution Neural Network In Clinical Decision Support System](https://airccj.org/CSCP/vol7/csit77505.pdf "Acute Leukemia Classification Using Convolution Neural Network In Clinical Decision Support System"), the authors propose a simple 5 layer Convolutional Neural Network. 

> "In this work, we proposed a network contains 4 layers. The first 3 layers for detecting features
> and the other two layers (Fully connected and Softmax) are for classifying the features. The input
> image has the size [50x50x3]. The receptive field (or the filter size) is 5x5. The stride is 1 then we move the filters one pixel at a time. The zero-padding is 2. It will allow us to control the spatial
> size of the output image (we will use it to exactly preserve the spatial size of the input volume so
> the input and output width and height are the same). During the experiment, we found that in our
> case, altering the size of original image during the convolution lead to decrease the accuracy
> about 40%. Thus the output image after convolution layer 1 has the same size with the input
> image."

> "The convolution layer 2 has the same structure with the convolution layer 1. The filter size is 5x5,
> the stride is 1 and the zero-padding is 2. The number of feature maps (the channel or the depth) in
> our case is 30. If the number of feature maps is lower or higher than 30, the accuracy will
> decrease 50%. By experiment, we found the accuracy also decrease 50% if we remove
> Convolution layer 2.""

> "The Max-Pooling layer 25x25 has Filter size is 2 and stride is 2. The fully connected layer has 2
> neural. Finally, we use the Softmax layer for the classification. "

In this project we will use an augmented dataset with the network proposed in this paper, built using Tensorflow 2.

We will build a Convolutional Neural Network, as shown in Fig 1, consisting of the following 5 layers (missing out the zero padding layers):

- Conv layer (50x50x30)
- Conv layer (50x50x30)
- Max-Pooling layer (25x25x30)
- Fully Connected layer (2 neurons)
- Softmax layer (Output 2)

&nbsp;

# Getting Started

To get started make sure you completed the steps on the [project home README](https://github.com/AMLResearchProject/ALL-Detection-System-2020 "project home README").

&nbsp;

# Data

Once you have your data you need to add it to the project filesystem. You will notice the data folder in the Model directory, **Model/Data**, inside you have two folders, **ALL-IDB-1** & **Test**. 

If you are using data augmentation, which this tutorial assumes, first take ten positive and ten negative samples and place them in the **Model/Data/Test** directory. This will be used by our Oculus Rift application and our testing purposes. In my case I used:

- im001_1.jpg
- im004_1.jpg
- im007_1.jpg
- im0011_1.jpg
- im0015_1.jpg
- im0018_1.jpg
- im0022_1.jpg
- im0026_1.jpg
- im0030_1.jpg
- im0033_1.jpg
- im0034_0.jpg
- im0037_0.jpg
- im0039_0.jpg
- im0042_0.jpg
- im0046_0.jpg
- im0065_0.jpg
- im0068_0.jpg
- im0076_0.jpg
- im0080_0.jpg

Next add the remaining 88 images to the **ALL-IDB-1** folder. The test images used in the demo will not be augmented, which I believe would be the case in a real world scenario.

Doing so decreased the metrics considerably, dropping from 97% to 95%, and an increased amount of false negatives. Increasing the amount of augmentation rotations per sample helped improve the model, but also increased false negatives.

&nbsp;

# Code structure

The code for this project consists of 4 main Python files and a configuration file:

- [config.json](hhttps://github.com/AMLResearchProject/ALL-Detection-System-2020/Classifier/Model/config.json "config.json"): The configuration file.
- [AllDS2020.py](hhttps://github.com/AMLResearchProject/ALL-Detection-System-2020/Classifier/AllDS2020.py "AllDS2020.py"): Core classifier wrapper class.
- [Helpers.py](hhttps://github.com/AMLResearchProject/ALL-Detection-System-2020/Classifier/Classes/Helpers.py "Helpers.py"): A helper class.
- [Data.py](hhttps://github.com/AMLResearchProject/ALL-Detection-System-2020/Classifier/Classes/Data.py "Data.py"): A data helpers class.
- [Model.py](hhttps://github.com/AMLResearchProject/ALL-Detection-System-2020/Classifier/Classes/Model.py "Model.py"): A model helpers class.

&nbsp;

## Classes 

Our functionality for this network can be found mainly in the **Classes** directory. 

|    Class | Description |
| ------------- | ------------ |
| Helpers.py   | [Helpers.py](hhttps://github.com/AMLResearchProject/ALL-Detection-System-2020/Classifier/Classes/Helpers.py "Helpers.py") is a helper class. The class loads the configuration and logging that the project uses.      |
| Data.py | [Data.py](hhttps://github.com/AMLResearchProject/ALL-Detection-System-2020/Classifier/Classes/Data.py "Data.py") is a data helper class. The class provides the functionality for sorting and preparing your training and validation data.  |     |
| Model.py | [Model.py](hhttps://github.com/AMLResearchProject/ALL-Detection-System-2020/Classifier/Classes/Model.py "Model.py") is a model helper class. The class provides the functionality for creating our CNN.       |

&nbsp;

### Functions

 The main functions are briefly explained below:

|    Class | Function |  Description |
| ------------- | ------------ | -------- |
| Data.py | data_and_labels_sort() | The data_and_labels_sort() function sorts the data into two Python lists, data[] and labels[]. |
| Data.py | data_and_labels_prepare() | The data_and_labels_prepare() function prepares the data and labels for training. |
| Data.py | convert_data() | The convert_data() function converts the training data to a numpy array. |
| Data.py | encode_labels() | The encode_labels() function One Hot Encodes the labels. |
| Data.py | shuffle() | The shuffle() function shuffles the data helping to eliminate bias. |
| Data.py | get_split() | The get_split() function splits the prepared data and labels into traiing and validation data. |
| Model.py | build_network() | The build_network() function creates the network architecture proposed in the [Acute Leukemia Classification Using Convolution Neural Network In Clinical Decision Support System](https://airccj.org/CSCP/vol7/csit77505.pdf "Acute Leukemia Classification Using Convolution Neural Network In Clinical Decision Support System") paper. |
| Model.py | compile_and_train() | The compile_and_train() function compiles and trains the model proposed in the [Acute Leukemia Classification Using Convolution Neural Network In Clinical Decision Support System](https://airccj.org/CSCP/vol7/csit77505.pdf "Acute Leukemia Classification Using Convolution Neural Network In Clinical Decision Support System") paper. |
| Model.py | evaluate_model() | The evaluate_model() function evaluates the model, and displays the values for the metrics we specified. |

&nbsp;

# Configuration

[config.json](hhttps://github.com/AMLResearchProject/ALL-Detection-System-2020/Classifier/Model/config.json "config.json")  holds the configuration for our network. 

```
{
    "cnn": {
        "data": {
            "dim": 50,
            "dim_augmentation": 100,
            "file_type": ".jpg",
            "rotations_augmentation": 25,
            "seed_adam": 32,
            "seed_adam_augmentation": 64,
            "seed_rmsprop": 3,
            "seed_rmsprop_augmentation": 6,
            "split": 0.255,
            "split_augmentation": 0.3,
            "train_dir": "Model/Data/ALL-IDB-1"
        },
        "train": {
            "batch": 80,
            "batch_augmentation": 100,
            "decay_adam": 1e-6,
            "decay_rmsprop": 1e-6,
            "epochs": 150,
            "epochs_augmentation": 150,
            "learning_rate_adam": 1e-4,
            "learning_rate_rmsprop": 1e-4,
            "val_steps": 10,
            "val_steps_augmentation": 3
        }
    }
}
```

We have the cnn object containing two objects, data and train. In data we have the configuration related to preparing the training and validation data. We use a seed to make sure our results are reproducible. In train we have the configuration related to training the model.

In my case, the configuration above was the best out of my testing, but you may find different configurations work better. Feel free to update these settings to your liking, and please let us know of your experiences.

&nbsp;

# Metrics

We can use metrics to measure the effectiveness of our model. In this network we will use the following metrics:

```
tf.keras.metrics.BinaryAccuracy(name='accuracy'),
tf.keras.metrics.Precision(name='precision'),
tf.keras.metrics.Recall(name='recall'),
tf.keras.metrics.AUC(name='auc')
```

These metrics will be displayed and plotted once our model is trained.  A useful tutorial while working on the metrics was the [Classification on imbalanced data](https://www.tensorflow.org/tutorials/structured_data/imbalanced_data) tutorial on Tensorflow's website.

&nbsp;

# Training the model

Now you are ready to train your model. As mentioned above, I used an Ubuntu machine with an NVIDIA GTX 1050 ti, using different machines/GPU/CPU may vary the results, if so please let us know your findings.

## Start The Training

Ensuring you have completed all previous steps, you can start training using the following command. 

```
python3 AllDS2020.py Train Adam  True
```

This tells the classifier to start in Train mode, use the Adam optimizer, and to to use data augmentation. You can play around and use rmsprop optimizer and no data augmentation etc, but for this tutorial the above is the command to use.

### Data

First the data will be prepared.

```
2020-03-05 12:26:43,952 - Data - INFO - All data: 88
2020-03-05 12:26:43,952 - Data - INFO - Positive data: 39
2020-03-05 12:26:43,952 - Data - INFO - Negative data: 49
2020-03-05 12:26:50,932 - Data - INFO - Data shape: (2904, 100, 100, 3)
2020-03-05 12:26:50,934 - Data - INFO - Labels shape: (2904, 2)
2020-03-05 12:26:50,934 - Data - INFO - Augmented data: (2904, 100, 100, 3)
2020-03-05 12:26:50,934 - Data - INFO - All Labels: (2904, 2)
2020-03-05 12:26:51,627 - Data - INFO - Training data: (2163, 100, 100, 3)
2020-03-05 12:26:51,627 - Data - INFO - Training labels: (2163, 2)
2020-03-05 12:26:51,627 - Data - INFO - Validation data: (741, 100, 100, 3)
2020-03-05 12:26:51,627 - Data - INFO - Validation labels: (741, 2)
```

### Model Summary

Our network matches the architecture proposed in the paper exactly, with exception to maybe the optimizer and loss function as this info was not provided in the paper. 

Before the model begins training, we will be shown the model summary, or architecture. 

```
Model: "AllCnn"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
zero_padding2d (ZeroPadding2 (None, 104, 104, 3)       0         
_________________________________________________________________
conv2d (Conv2D)              (None, 100, 100, 30)      2280      
_________________________________________________________________
zero_padding2d_1 (ZeroPaddin (None, 104, 104, 30)      0         
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 100, 100, 30)      22530     
_________________________________________________________________
max_pooling2d (MaxPooling2D) (None, 50, 50, 30)        0         
_________________________________________________________________
flatten (Flatten)            (None, 75000)             0         
_________________________________________________________________
dense (Dense)                (None, 2)                 150002    
_________________________________________________________________
activation (Activation)      (None, 2)                 0         
=================================================================
Total params: 174,812
Trainable params: 174,812
Non-trainable params: 0
```

## Metics

<img src="Media/Images/Metrics.png" alt="Adam Optimizer Results" />

_Fig 2. Ubuntu/GTX 1050 ti Adam Optimizer Results_

Here you can find the training results that are shown in console during training.

```
2020-03-05 12:43:03,292 - Model - INFO - Metrics: loss 0.0797744454733032
2020-03-05 12:43:03,293 - Model - INFO - Metrics: acc 0.97840756
2020-03-05 12:43:03,293 - Model - INFO - Metrics: precision 0.97840756
2020-03-05 12:43:03,293 - Model - INFO - Metrics: recall 0.97840756
2020-03-05 12:43:03,293 - Model - INFO - Metrics: auc 0.995607

2020-03-05 12:43:03,607 - Model - INFO - Confusion Matrix: [[395   0]
 [ 16 330]]

2020-03-05 12:47:07,458 - Model - INFO - True Positives: 330(44.534412955465584%)
2020-03-05 12:47:07,504 - Model - INFO - False Positives: 0(0.0%)
2020-03-05 12:47:07,504 - Model - INFO - True Negatives: 395(53.30634278002699%)
2020-03-05 12:47:07,505 - Model - INFO - False Negatives: 16(2.1592442645074224%)
2020-03-05 12:47:07,505 - Model - INFO - Specificity: 1.0
2020-03-05 12:47:07,505 - Model - INFO - Misclassification: 16(2.1592442645074224%)
```

&nbsp;

# Contributing

The Peter Moss Acute Myeloid & Lymphoblastic Leukemia AI Research project encourages and welcomes code contributions, bug fixes and enhancements from the Github.

Please read the [CONTRIBUTING](https://github.com/AMLResearchProject/ALL-Detection-System-2020/blob/master/CONTRIBUTING.md "CONTRIBUTING") document for a full guide to forking our repositories and submitting your pull requests. You will also find information about our code of conduct on this page.

## Contributors

- [Adam Milton-Barker](https://www.leukemiaresearchassociation.ai/ "Adam Milton-Barker") - [Peter Moss Leukemia AI Research](https://www.leukemiaresearchassociation.ai "Peter Moss Leukemia AI Research") & Intel Software Innovator, Sabadell, Spain

&nbsp;

# Versioning

We use SemVer for versioning. For the versions available, see [Releases](https://github.com/AMLResearchProject/ALL-Detection-System-2020/releases "Releases").

# License

This project is licensed under the **MIT License** - see the [LICENSE](https://github.com/AMLResearchProject/ALL-Detection-System-2020/blob/master/LICENSE "LICENSE") file for details.

# Bugs/Issues

We use the [repo issues](https://github.com/AMLResearchProject/ALL-Detection-System-2020/issues "repo issues") to track bugs and general requests related to using this project. See [CONTRIBUTING](https://github.com/AMLResearchProject/ALL-Detection-System-2020/blob/master/CONTRIBUTING.md "CONTRIBUTING") for more info on how to submit bugs, feature requests and proposals.
