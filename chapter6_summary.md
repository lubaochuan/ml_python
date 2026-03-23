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
