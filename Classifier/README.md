# Peter Moss Acute Myeloid & Lymphoblastic Leukemia AI Research Project

## Acute Lymphoblastic Leukemia Detection System 2020

&nbsp;

# AllDS2020 Classifier

## Introduction

This project is the classifier that is used in Acute the Lymphoblastic Leukemia Detection System 2020. The network provided in this project was originally created in my [ALL research papers evaluation project](https://github.com/leukemiaresearchassociation/ALL-IDB-Classifiers/blob/master/Python/Tensorflow/2-0/Classification/ALL-Papers/Evaluations/Paper-1.md "ALL research papers evaluation project"), where I replicated the network proposed in the [Acute Leukemia Classification Using Convolution Neural Network In Clinical Decision Support System](https://airccj.org/CSCP/vol7/csit77505.pdf "Acute Leukemia Classification Using Convolution Neural Network In Clinical Decision Support System") paper by Thanh.TTP, Giao N. Pham, Jin-Hyeok Park, Kwang-Seok Moon, Suk-Hwan Lee, and Ki-Ryong Kwon, and the data augmentation proposed in  [Leukemia Blood Cell Image Classification Using Convolutional Neural Network](http://www.ijcte.org/vol10/1198-H0012.pdf "Leukemia Blood Cell Image Classification Using Convolutional Neural Network") by T. T. P. Thanh, Caleb Vununu, Sukhrob Atoev, Suk-Hwan Lee, and Ki-Ryong Kwon. The original project was inspired by the [work](https://github.com/AmlResearchProject/AML-ALL-Classifiers/blob/master/Python/_Keras/QuantisedCode/QuantisedCode.ipynb "work") done by [Amita Kapoor](https://www.leukemiaresearchassociation.ai/team/amita-kapoor/profile "Amita Kapoor") and [Taru Jain](https://www.leukemiaresearchassociation.ai/student-program/student/taru-jain "Taru Jain") and my previous [projects](https://github.com/AMLResearchProject/AML-ALL-Classifiers/tree/master/Python/_Keras/AllCNN "projects") based on their work. 

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

&nbsp;

## Getting Started

To get started make sure you completed the steps on the [project home README](https://github.com/AMLResearchProject/ALL-Detection-System-2020 "project home README").

&nbsp;

# Contributing

The Peter Moss Acute Myeloid & Lymphoblastic Leukemia AI Research project encourages and welcomes code contributions, bug fixes and enhancements from the Github.

Please read the [CONTRIBUTING](https://github.com/AMLResearchProject/ALL-Detection-System-2020/blob/master/CONTRIBUTING.md "CONTRIBUTING") document for a full guide to forking our repositories and submitting your pull requests. You will also find information about our code of conduct on this page.

## Contributors

- [Adam Milton-Barker](https://www.leukemiaresearchassociation.ai/ "Adam Milton-Barker") - [Peter Moss Leukemia AI Research](https://www.leukemiaresearchassociation.ai "Peter Moss Leukemia AI Researchr") & Intel Software Innovator, Barcelona, Spain

&nbsp;

# Versioning

We use SemVer for versioning. For the versions available, see [Releases](https://github.com/AMLResearchProject/ALL-Detection-System-2020/releases "Releases").

# License

This project is licensed under the **MIT License** - see the [LICENSE](https://github.com/AMLResearchProject/ALL-Detection-System-2020/blob/master/LICENSE "LICENSE") file for details.

# Bugs/Issues

We use the [repo issues](https://github.com/AMLResearchProject/ALL-Detection-System-2020/issues "repo issues") to track bugs and general requests related to using this project. See [CONTRIBUTING](https://github.com/AMLResearchProject/ALL-Detection-System-2020/blob/master/CONTRIBUTING.md "CONTRIBUTING") for more info on how to submit bugs, feature requests and proposals.
