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

### Requirements

| Package | Version |
| :- | -:|
| NumPy | 1.18 |
| TensorFlow | 2.2.0 |
| TensorFlow Datasets | 2.1.0 |
| TensorBoard | 2.2.0 |

## Data

### Choose dataset 

* [MNIST](http://yann.lecun.com/exdb/mnist/) (Modified National Institute of Standards and Technology)

### Load dataset

### Explore dataset

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

### Compile model

#### Loss function

![Cross Entropy](https://render.githubusercontent.com/render/math?math=L_%7BCE%7D(p)%3D%20%5Clog(p)%20%2B%20%5Clog(1-p))

#### Objective function

![Discriminator](https://render.githubusercontent.com/render/math?math=%5Cmax_%7BD%7D%5C%2CE_%7Bx%5Csim%20p_%7Bdata%7D%7D%5C%2C%5Clog(D%5Bx%5D))

![Generator](https://render.githubusercontent.com/render/math?math=%5Cmin_%7BG%7D%5C%2CE_%7Bz%5Csim%20p_%7Bnoise%7D%7D%5C%2C%5Clog(1-D%5BG(z)%5D))

![GAN](https://render.githubusercontent.com/render/math?math=%5Cmin_%7BG%7D%5C%2C%5Cmax_%7BD%7D%5C%2CE_%7Bx%5Csim%20p_%7Bdata%7D%7D%5C%2C%5Clog(D%5Bx%5D)%20%2B%20%5C%2CE_%7Bz%5Csim%20p_%7Bnoise%7D%7D%5C%2C%5Clog(1-D%5BG(z)%5D))

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

### Export model

### Export metadata
