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

* Update weights: $w \leftarrow w - \eta \nabla L$

## 6. Backpropagation Intuition

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

# Review Questions

### 1. Why do neural networks need nonlinear activation functions?

<details>
<summary>Answer</summary>

Without nonlinear activation functions, each layer performs only a linear transformation. Composing multiple linear layers still results in a **single linear function**, meaning the network cannot learn complex patterns. Nonlinearity allows the network to approximate **complex, nonlinear relationships**.
</details>

### 2. What is the difference between a perceptron and a multi-layer perceptron?

<details>
<summary>Answer</summary>

* **Perceptron**:

  * Single layer
  * Can only learn **linearly separable** problems
* **Multi-Layer Perceptron (MLP)**:

  * Multiple layers (hidden layers)
  * Can learn **nonlinear decision boundaries**
  * Much more expressive
</details>

### 3. What problem does backpropagation solve?

<details>
<summary>Answer</summary>

Backpropagation efficiently computes the **gradient of the loss function with respect to each weight** in the network.
This allows the model to update weights using gradient descent and **learn from errors**.
</details>

### 4. Why is ReLU preferred over sigmoid in many cases?

<details>
<summary>Answer</summary>

* ReLU:

  * Avoids vanishing gradient for positive inputs
  * Computationally efficient
  * Leads to faster training
* Sigmoid:

  * Saturates at extreme values → gradients become very small
  * Slows down learning
</details>

### 5. What happens if the learning rate is too high? Too low?

<details>
<summary>Answer</summary>

* **Too high**:

  * Training becomes unstable
  * Loss may oscillate or diverge
* **Too low**:

  * Training is very slow
  * May get stuck in local minima
</details>

### 6. Given a network output, how is the loss computed?

<details>
<summary>Answer</summary>

The loss function compares the **predicted output** with the **true label**:

* Regression → Mean Squared Error (MSE)
* Classification → Cross-Entropy

Example:
$$
\text{Loss} = (y_{\text{true}} - y_{\text{pred}})^2
$$
</details>

### 7. How does mini-batch gradient descent balance efficiency and stability?

<details>
<summary>Answer</summary>

* Uses small batches of data:

  * Faster than full batch (less computation per step)
  * More stable than stochastic (less noisy updates)
* Provides a **good trade-off between speed and convergence quality**
</details>

### 8. Why is softmax used in multi-class classification?

<details>
<summary>Answer</summary>

Softmax converts raw outputs (logits) into **probabilities that sum to 1**, making it suitable for:

* Multi-class prediction
* Interpretable outputs (confidence levels)
</details>

### 9. What are signs of overfitting in training?

<details>
<summary>Answer</summary>

Model memorizes training data but fails to generalize
* Training accuracy is high
* Validation accuracy is low
* Model performs poorly on new data
</details>

### 10. How can you improve a poorly performing neural network?

<details>
<summary>Answer</summary>

Possible improvements:

* Adjust architecture (more/less layers)
* Tune learning rate
* Add more data
* Use regularization (dropout, L2)
* Normalize input data
* Train longer (more epochs)
</details>

### 11. What does this layer do?

```python
keras.layers.Dense(10, activation="softmax")
```

<details>
<summary>Answer</summary>

* Fully connected layer with **10 neurons**
* Applies **softmax activation**
* Outputs **probabilities for 10 classes**
</details>

### 12. What is the role of `model.compile()`?

<details>
<summary>Answer</summary>

It configures the model for training by specifying:

* Loss function
* Optimizer (e.g., SGD)
* Metrics (e.g., accuracy)
</details>

### 13. What happens during `model.fit()`?

<details>
<summary>Answer</summary>

* Runs the training loop:

  1. Forward pass
  2. Compute loss
  3. Backpropagation
  4. Update weights
* Repeats for multiple epochs
</details>

### 14. Why can a neural network with no activation functions only learn linear relationships?

<details>
<summary>Answer</summary>

Because each layer computes a linear transformation:

$$
y = W_2(W_1x + b_1) + b_2
$$

This simplifies to:

$$
y = Wx + b
$$

So the entire network behaves like a **single linear model**, regardless of depth.
</details>


