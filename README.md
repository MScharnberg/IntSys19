# IntSys19

## Contents

* Data
  * Choose dataset 
  * Load dataset
  * Explore dataset
  * Preprocess dataset
* Model
  * Choose model 
  * Compile model
  * Train model
  * Evaluate model
* Deployment
  * Use model
  * Export model

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

## Model

### Choose model 

* [GAN](https://arxiv.org/abs/1406.2661) (Generative Adversarial Network)

![Architecture](./img/GAN.png)

D: Discriminator, G: Generator | p_real/p_noise: Real/Fake data distribution | x: Original, x': Copy | z: Noise

### Compile model

* Loss
  * Binary Crossentropy
* Optimizer
  * Adam (Adaptive Moment Estimation)
  
### Train model

### Evaluate model

## Deployment 

### Use model

### Export model
