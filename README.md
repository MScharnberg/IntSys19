# IntSys19

## Contents

* [Data](https://github.com/MScharnberg/IntSys19/tree/documentation#data)
  * Choose dataset 
  * Load dataset
  * Explore dataset
  * Preprocess dataset
  * Visualize dataset
* [Model](https://github.com/MScharnberg/IntSys19/tree/documentation#model)
  * Choose model 
  * Compile model
  * Train model
  * Evaluate model
* [Deployment](https://github.com/MScharnberg/IntSys19/tree/documentation#deployment)
  * Use model
  * Export model
  * Export metadata

## Setup

### Getting started

OPTION 1: Open [Notebook](#) in [CoLab](#)

OPTION 2: Install requirements via `pip install -r requirements.txt` and run [Script](#) on local machine.

### Requirements

| Package             | Version |
|:--------------------|--------:|
| NumPy               | 1.18    |
| TensorFlow          | 2.2.0   |
| TensorFlow Datasets | 2.1.0   |
| TensorBoard         | 2.2.0   |

## Data

### Choose dataset 

* [MNIST](http://yann.lecun.com/exdb/mnist/) (Modified National Institute of Standards and Technology)

### Load dataset

### Explore dataset

![Real](#)

Real digits

### Preprocess dataset

### Visualize dataset

## Model

### Choose model 

* [GAN](https://arxiv.org/abs/1406.2661) (Generative Adversarial Network)
  * Generator as generative model
    * Map imaginary to real data distribution
    * Copy digits as close to original ones as regression problem
  * Discriminator as discriminative model
    * Distinguish imaginary from real data distribution
    * Differentiate between a digit's copy and original as classification problem

![Architecture](./img/GAN.png)

D: Discriminator, G: Generator | p_real/p_noise: Real/Fake data distribution | x: Original, x': Copy | z: Noise

### Explore model

![Generator](#)

Generator

![Discriminator](#)

Discriminator

### Compile model

* Loss function
  * Discriminator
  * Generator
  * Discriminator + Generator = GAN

* Loss metrics
  * Generator
  * Discriminator
  * Real vs. Fake
  
* Optimizer
  * [Adam](https://arxiv.org/abs/1412.6980) (Adaptive Moment Estimation)

### Train model

### Evaluate model

* Evaluation metrics
  * MSE (Mean-squared error)
  * KLD (Kullback–Leibler divergence)

## Deployment 

### Use model

![Fake](#)

Fake digits

### Export model

### Export metadata