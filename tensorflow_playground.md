# TensorFlow Playground

This TensorFlow Playground exercise explores how hidden layers and neurons affect decision boundary complexity.

## Part 0 — Setup

1. Open: [https://playground.tensorflow.org/](https://playground.tensorflow.org/)
2. Select dataset: **XOR**
3. Features: Only `x1`, `x2`
4. Activation: `tanh`
5. Learning rate: default (≈0.03)
6. Noise: 0

> “Can a straight line separate this data?”

## Part 1 — No Hidden Layers (Linear Model)

### Instructions

* Remove all hidden layers (0 layers)
* Click **Run**

### Observe

* Decision boundary is a **straight line**
* Model fails to separate classes

### Reflection Questions

1. What shape is the boundary?
2. Why does it fail on XOR?

Insight:

* This is essentially logistic regression → only **linear boundaries**

## Part 2 — One Hidden Layer (Start of Non-Linearity)

### Instructions

* Add **1 hidden layer**
* Set **2 neurons**

### Observe

* Boundary becomes **slightly curved**
* Still not perfect

### Now increase neurons:

* Try: 4 → 6 → 8 neurons

### Reflection Questions

1. How does the boundary change as neurons increase?
2. What do extra neurons “add”?

<details>
<summary>Answer</summary>

Each neuron adds a “cut” in space → more neurons = more pieces ([Data Science Stack Exchange][3])
</details>

## Part 3 — Two Hidden Layers (Depth Matters)

### Instructions

* Add a **second hidden layer**
* Try:

  * Layer 1: 4 neurons
  * Layer 2: 2 neurons

### Observe

* Boundary becomes **more flexible and smoother**
* Model solves XOR well

### Reflection Questions

1. How is this different from just adding more neurons to one layer?
2. Does depth change *how* the model learns?

<details>
<summary>Answer</summary>

Layers build **hierarchical features** (simple → complex)
</details>

## Part 4 — Challenge: Circles Dataset

### Instructions

* Switch dataset → **Circle**
* Try configurations:

| Model | Layers   | Neurons   |
| ----- | -------- | --------- |
| A     | 1 layer  | 2 neurons |
| B     | 1 layer  | 8 neurons |
| C     | 2 layers | 4 + 2     |
| D     | 3 layers | 6 + 4 + 2 |

### Observe

* A: fails (too simple)
* B: jagged boundary
* C/D: smooth circular boundary

### Reflection Questions

1. Which model best fits the circle?
2. Which model looks “overly complicated”?

<details>
<summary>Answer</summary>

* Depth helps form **smooth, structured boundaries**
* Width alone can create **patchy approximations**
</details>

## Part 5 — Hard Mode: Spiral Dataset 🌪️

### Instructions

* Switch dataset → **Spiral**
* Start with: 1 layer, 4 neurons → FAIL
* Gradually increase:
  * 2 layers, 6 neurons each
  * 3 layers, varied neurons

### Observe

* Only deeper networks can capture spiral shape

### Reflection Questions

1. Why does shallow fail here?
2. What does the network need to “build” the spiral?

<details>
<summary>Answer</summary>

Complex patterns require **multiple transformations stacked together**
</details>

# Takeaways

### 1. Width (neurons per layer)

* Adds more **cuts/regions**
* Can approximate complexity, but often messy

### 2. Depth (number of layers)

* Builds **hierarchical features**
* Enables smoother, more structured boundaries

### 3. Decision Boundaries

* Linear model → straight line
* Shallow network → piecewise curves
* Deep network → complex, smooth shapes
