# numpy-cnn-from-scratch

Implementation of a Convolutional Neural Network from scratch using only NumPy.
> 🚀 Building a trainable CNN engine from scratch — updates coming soon.


## Title

### CNN From Scratch with NumPy

A fully working Convolutional Neural Network implemented only with NumPy.  
No TensorFlow, no PyTorch — just pure math and array operations.


## Overview

This project implements a complete CNN engine from scratch using only NumPy.  
The goal is to understand the internal mechanics of deep learning by building  
every component manually — convolution, activation functions, pooling,  
fully-connected layers, softmax, and even backpropagation.

This is not a toy example using a single forward pass.  
This is a **trainable**, **end-to-end** deep learning engine.


## Features

- Pure NumPy implementation (no ML frameworks)
- Conv2D layer (padding, stride)
- ReLU activation
- MaxPooling layer
- Fully Connected (dense) layer
- Softmax + Cross Entropy Loss
- Backpropagation implemented manually
- Stochastic Gradient Descent optimizer
- Works on MNIST or CIFAR-10


## Architecture

- conv/
  - conv2d.py
  - padding.py
  - im2col.py (optional)
- layers/
  - relu.py
  - maxpool.py
  - dense.py
- core/
  - model.py
  - losses.py
  - optimizer.py
  - utils.py
- train.py
- inference_example.py


## How It Works

Each layer exposes the following interface:

- forward(x): computes output
- backward(dout): propagates gradients backward
- update(lr): updates internal parameters (if any)

The entire network is trained using standard SGD.

Backpropagation is implemented using vectorized NumPy operations  
to match the logic used inside real deep learning frameworks.


## Getting Started

### Install

```bash
pip install -r requirements.txt
```

### Train on MNIST

```bash
python train.py --datasetmnist --epochs 3
```

### Run inference

```bash
python inference_example.py
```


## Results

- MNIST accuracy:
- Training time:


## Why This Project Exists

- Deep learning frameworks hide the internal math behind abstractions.
- Building a CNN from scratch is the fastest way to truly understand:
  - convolution mechanics
  - gradient flow
  - how layers interact
  - why training works
  - what tensor frameworks actually do internally

- This project is meant to be a bridge between theory and engineering.


## Future Work

- Add momentum optimizer
- Add batch normalization
- Add dropout
- Build a FastAPI inference server


## License

MIT License


