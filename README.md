# IntSys19

## Contents

* Data
  * Choose dataset 
  * Load dataset
  * Explore dataset
  * Preprocess dataset
  * Visualize dataset
* Model
  * Choose model 
  * Compile model
  * Train model
  * Evaluate model
* Deployment
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

![Architecture](./img/GAN.png)

D: Discriminator, G: Generator | p_real/p_noise: Real/Fake data distribution | x: Original, x': Copy | z: Noise

### Compile model

#### Loss function

![Cross Entropy](https://render.githubusercontent.com/render/math?math=L_%7BCE%7D(p)%3D%20%5Clog(p)%20%2B%20%5Clog(1-p))

#### Objective function

![Discriminator](https://render.githubusercontent.com/render/math?math=%5Cmax_%7BD%7D%20E_%7Bx%20%5Csim%20p_%7Bdata%7D%7D%20%5B%5Clog%20D(x)%5D)

![Generator](https://render.githubusercontent.com/render/math?math=%5Cmin_%7BG%7D%20E_%7Bz%20%5Csim%20p_%7Bnoise%7D%7D%20%5B%5Clog%201-D(G(z))%5D)

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
