# Motion artifact correction by applying a novel unsupervised network for arterial phase imaging of gadoxetic acid-enhanced liver MRI examinations

## Introduction

This is an implementation for the paper **"Motion artifact correction by applying a novel unsupervised network for arterial phase imaging of gadoxetic acid-enhanced liver MRI examinations"**, a simple and efficient framework for unsupervised MRI motion correction, which is injected into the general domain transfer architecture. More details could be found in the original paper.
 
## Prerequisites

* (OS) Windows/Ubuntu
* Python >= 3.6
* Pytorch >= 1.1.0
* Python-Libs, e.g., cv2, skimage.

## Training

* Prepare your dataset. 
* Update the data paths in `config.py` and `utils.py` file.
* Train your model by the `train.py` file.

## Test

A simple script to test your model:

`python3 test.py`

## Acknowledge

Our code is based on the [LIR-for-Unsupervised-IR](https://github.com/Wenchao-Du/LIR-for-Unsupervised-IR), which is a nice work for unsupervised image translation.

 
