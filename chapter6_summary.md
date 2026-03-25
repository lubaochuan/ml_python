# Chapter 6 Summary: Decision Trees

Decision Trees are a **supervised learning algorithm** used for both:

* **Classification**
* **Regression**

They work by **recursively splitting the dataset** into subsets based on feature values, forming a tree-like structure of decisions.

## Intuition

A Decision Tree asks a sequence of **if–then questions**:

```text
Is feature ≤ threshold?
    ├── Yes → go left
    └── No  → go right
```

The goal is to **partition the data** so that each leaf node is as “pure” as possible.

## Structure of a Decision Tree

* **Root Node**: Top of the tree (first split)
* **Internal Nodes**: Decision rules (feature tests)
* **Leaf Nodes**: Final predictions
* **Branches**: Outcomes of decisions

## How Trees Are Built (CART Algorithm)

Most libraries (like Scikit-Learn) use **CART (Classification and Regression Trees)**.

### Step-by-step:

1. Try all possible splits across all features
2. Measure **impurity reduction**
3. Choose the split that produces the **purest subsets**
4. Repeat recursively

## Impurity Measures

### 1. Gini Impurity (default in Scikit-Learn)

$$
Gini = 1 - \sum_{k=1}^{K} p_k^2
$$

* Measures how often a randomly chosen element would be misclassified
* Lower = better (more pure)

### 2. Entropy (Information Gain)

$$
Entropy = -\sum_{k=1}^{K} p_k \log_2(p_k)
$$

* Measures disorder or uncertainty
* Used in ID3/C4.5 algorithms

## Splitting Criteria

* Choose split that **maximizes impurity reduction**
* Also called:

  * **Information Gain** (entropy-based)
  * **Gini reduction**

## Decision Tree for Classification

* Output = **predicted class**
* Leaf node stores:

  * Class distribution
  * Predicted class (majority vote)

## Decision Tree for Regression

* Output = **numerical value**
* Splits minimize **Mean Squared Error (MSE)**

$$
MSE = \frac{1}{n} \sum (y_i - \hat{y})^2
$$

## Overfitting Problem

Decision Trees can easily **overfit**:

* Very deep trees memorize training data
* Poor generalization

## Regularization Techniques

Control tree complexity using:

* `max_depth`
* `min_samples_split`
* `min_samples_leaf`
* `max_leaf_nodes`

Goal: **balance bias vs variance**

## Pruning

### Types:

* **Pre-pruning**: Stop early (limit depth, etc.)
* **Post-pruning**: Grow full tree, then trim

Scikit-Learn supports:

* **Cost Complexity Pruning (`ccp_alpha`)**

## Decision Boundaries

* Trees create **axis-aligned splits**
* Resulting boundaries are:

  * Rectangular
  * Not smooth

This is a limitation compared to models like SVM.

## Feature Importance

Decision Trees can estimate:

* **How important each feature is**

Based on:

* Total impurity reduction contributed by that feature

## Advantages

* Easy to understand and interpret
* Requires little data preprocessing
* Handles numerical and categorical data
* Nonlinear relationships

## Disadvantages

* Prone to overfitting
* Sensitive to small data changes
* Axis-aligned splits (limited flexibility)
* Poor performance compared to ensembles

## From Trees to Forests

Single trees are weak → leads to:

* **Random Forests**
* **Gradient Boosted Trees**

These improve performance significantly

# Glossary

### Decision Tree

A model that splits data using feature-based decisions to make predictions.

### Leaf Node

A terminal node that outputs a prediction.

### Internal Node

A node that represents a decision rule (feature split).

### Gini Impurity

A measure of how mixed the classes are in a node.

### Entropy

A measure of disorder or uncertainty in a dataset.

### Information Gain

Reduction in entropy after a split.

### Split

Dividing data based on a feature and threshold.

### CART Algorithm

A greedy algorithm that builds binary trees using impurity reduction.

### Regression Tree

A decision tree that predicts continuous values.

### Overfitting

When a model memorizes training data instead of generalizing.

### Regularization

Techniques to limit model complexity and improve generalization.

### Pruning

Removing parts of a tree to reduce overfitting.

### Feature Importance

A score indicating how useful a feature is in making predictions.

### Decision Boundary

The regions in feature space where predictions change.

### Ensemble Methods

Techniques that combine multiple models (e.g., Random Forests).

# Key Takeaways

* Decision Trees are **intuitive and powerful**, but:

  * Easily overfit
  * Require careful tuning
* They form the foundation of:

  * **Random Forests**
  * **Gradient Boosting**

# Review Questions

## Level 1: Basic Understanding (Warm-up)

### Q1.

You train a decision tree and it achieves **100% accuracy on training data** but performs poorly on test data.

What is the most likely issue?

a) Underfitting
b) Overfitting
c) Data leakage
d) High bias

### Q2.

A node contains:

* 50% Class A
* 50% Class B

What can you say about its impurity?

a) Very low
b) Medium
c) Very high
d) Zero

### Q3.

A split produces two child nodes:

* Left: 90% A, 10% B
* Right: 85% A, 15% B

What is the best interpretation?

a) Excellent split
b) Useless split
c) Moderate improvement
d) Perfect classification

## Level 2: Think Like the Algorithm

### Q4.

Why does a decision tree choose splits **greedily** (best split at current step)?

a) It guarantees the global optimum
b) It reduces computational cost
c) It improves interpretability
d) It avoids overfitting

### Q5.

You have two features:

* Feature X → gives a small impurity reduction early
* Feature Y → gives a large reduction later (but not chosen first)

Why might the tree miss the better overall structure?

a) Trees are random
b) Trees are greedy
c) Trees only use one feature
d) Trees ignore deeper splits

### Q6.

A feature appears near the **root** of the tree.

What does this usually indicate?

a) It has missing values
b) It is highly predictive
c) It is categorical
d) It has low variance

## Level 3: What-If Scenarios

### Q7.

If you increase `max_depth`, what happens?

a) Bias increases, variance decreases
b) Bias decreases, variance increases
c) Both increase
d) Both decrease

### Q8.

If you set `min_samples_leaf = 50`, what happens?

a) Tree becomes deeper
b) Tree becomes simpler
c) Tree becomes random
d) Tree stops splitting entirely

### Q9.

What happens if your dataset has a lot of **noise** and you don’t regularize the tree?

a) Tree becomes shallow
b) Tree ignores noise
c) Tree overfits noise
d) Tree becomes linear

## Level 4: Decision Boundary Intuition

### Q10.

Decision trees create what kind of decision boundaries?

a) Smooth curves
b) Circular regions
c) Axis-aligned rectangles
d) Random shapes

### Q11.

Why might a decision tree struggle with diagonal patterns?

a) It cannot split data
b) It only makes axis-aligned splits
c) It ignores features
d) It uses randomness

## Level 5: Debugging Intuition

### Q12.

You slightly change one data point, and the tree structure changes drastically.

Why?

a) Trees are stable models
b) Trees are sensitive to data
c) Trees ignore small changes
d) Trees use averaging

### Q13.

Your model performs poorly. You notice:

* Tree is very shallow (depth = 2)

What is the likely issue?

a) Overfitting
b) Underfitting
c) Data leakage
d) Too much data

### Q14.

You compare:

* Model A: single decision tree
* Model B: random forest

Model B performs better. Why?

a) It uses deeper trees
b) It reduces variance
c) It reduces bias only
d) It uses fewer features

## Challenge Level (Deep Intuition)

### Q15.

You have a perfectly separable dataset, but your tree is not achieving perfect accuracy.

What could be the reason?

a) max_depth too small
b) impurity measure is wrong
c) dataset too large
d) too many features

### Q16.

Which situation makes decision trees perform poorly?

a) Clear feature thresholds
b) Nonlinear relationships
c) Smooth continuous relationships
d) Small datasets

### Q17.

Why are decision trees often combined into ensembles?

a) To increase interpretability
b) To reduce variance and improve generalization
c) To simplify models
d) To remove features

<details>
<summary>
Answer Key
</summary>

**Level 1**
Q1: b
Q2: c
Q3: c

**Level 2**
Q4: b
Q5: b
Q6: b

**Level 3**
Q7: b
Q8: b
Q9: c

**Level 4**
Q10: c
Q11: b

**Level 5**
Q12: b
Q13: b
Q14: b

**Challenge**
Q15: a
Q16: c
Q17: b

</details>
