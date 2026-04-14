# Chapter 10 Summary: Introduction to Artificial Neural Networks with Keras

Artificial Neural Networks (ANNs) are powerful models inspired by the brain, capable of learning complex patterns from data. This chapter introduces:

* The **core intuition** behind neural networks
* The **perceptron and multi-layer perceptrons (MLPs)**
* How **training works via backpropagation**
* Practical implementation using **Keras deep learning library**

## 1. From Biological Neurons to Artificial Neurons

### Biological Inspiration

* Neurons receive signals → process → transmit output
* Learning occurs by strengthening/weakening connections

### Artificial Neuron (Simplified)

* Inputs: $(x_1, x_2, ..., x_n)$

* Weights: $(w_1, w_2, ..., w_n)$

* Output:

  $z = \sum w_i x_i + b$

  $\text{output} = \phi(z)$

* $(\phi)$: activation function

## 2. The Perceptron

### Key Idea

* A **linear classifier**
* Computes a weighted sum and applies a step function

### Limitations

* Can only learn **linearly separable problems**
* Cannot solve XOR

## 3. Multi-Layer Perceptrons (MLPs)

### Structure

* Input layer
* One or more **hidden layers**
* Output layer

### Key Insight

* Adding hidden layers allows:

  * Nonlinear decision boundaries
  * Learning complex patterns

### Universal Approximation

* MLPs can approximate **any continuous function** (given enough neurons)

## 4. Activation Functions

### Why Needed?

Without activation → network becomes just a linear model.

### Common Functions

| Function | Formula                   | Use Case                           |
| -------- | ------------------------- | ---------------------------------- |
| Sigmoid  | $\frac{1}{1 + e^{-z}}$  | Binary classification (historical) |
| Tanh     | $\tanh(z)$            | Zero-centered                      |
| ReLU     | $\max(0, z)$           | Most common                        |
| Softmax  | Converts to probabilities | Multi-class output                 |

## 5. Training Neural Networks

### Step 1: Forward Pass

* Compute predictions layer by layer

### Step 2: Loss Computation

* Measure error (e.g., MSE, cross-entropy)

### Step 3: Backpropagation

* Compute gradients using chain rule

### Step 4: Gradient Descent

* Update weights:

$w \leftarrow w - \eta \nabla L$

## ⚡ 6. Backpropagation Intuition

* Error flows **backward** through the network
* Each weight updated based on:

  * Its contribution to the error
  * Learning rate

Key idea: “Blame assignment” — which weights caused the error?

## 7. Implementing MLPs with Keras

### Typical Workflow

```python
from tensorflow import keras

model = keras.models.Sequential([
    keras.layers.Dense(30, activation="relu"),
    keras.layers.Dense(10, activation="softmax")
])

model.compile(
    loss="sparse_categorical_crossentropy",
    optimizer="sgd",
    metrics=["accuracy"]
)

model.fit(X_train, y_train, epochs=10)
```

### Key Components

* `Dense`: fully connected layer
* `compile`: define loss + optimizer
* `fit`: training loop

## 8. Regression vs Classification

### Regression

* Output: continuous value
* No activation or linear output
* Loss: MSE

### Classification

* Output: probability
* Activation: sigmoid / softmax
* Loss: cross-entropy

## 9. Common Challenges

### Vanishing/Exploding Gradients

* Gradients become too small or large
* Hard to train deep networks

### Overfitting

* Model memorizes training data
* Solutions:

  * Regularization
  * Dropout
  * More data

### Initialization Matters

* Poor initialization slows or breaks training

## 10. Key Takeaways

* Neural networks learn **hierarchical representations**
* Depth → more expressive power
* Backpropagation enables efficient training
* Activation functions introduce **nonlinearity**
* Keras simplifies building and training models

# Glossary of Key Terms

### Artificial Neural Network (ANN)

A model composed of layers of interconnected neurons that learn patterns from data.

### Perceptron

A single-layer neural network used for binary classification.

### Multi-Layer Perceptron (MLP)

A neural network with one or more hidden layers.

### Weight

A parameter that determines the importance of an input.

### Bias

A constant added to shift the activation function.

### Activation Function

A function that introduces nonlinearity (e.g., ReLU, sigmoid).

### ReLU (Rectified Linear Unit)

Activation function defined as $\max(0, z)$.

### Softmax

Function that converts outputs into probabilities.

### Loss Function

Measures how far predictions are from actual values.

### Gradient Descent

Optimization algorithm that minimizes loss by updating parameters.

### Backpropagation

Algorithm to compute gradients efficiently using the chain rule.

### Epoch

One full pass through the training dataset.

### Learning Rate

Step size used when updating weights.

### Overfitting

When a model performs well on training data but poorly on new data.

### Underfitting

When a model is too simple to capture patterns.

### Hidden Layer

Intermediate layer between input and output.

## Tips for **visual + intuitive learning**:

* Decision boundary evolution
* Layer-by-layer transformation
* “Feature extraction” perspective:

  * Early layers → simple patterns
  * Later layers → complex abstractions
