# Chapter 6 Summary: Decision Trees

Decision Trees are intuitive, interpretable models that:

* Split data based on feature values
* Create a tree of decisions
* Can handle both **classification** and **regression**

They mimic human decision-making (“if–then” rules)

## 1. How Decision Trees Work

A decision tree:

* Starts at the **root node**
* Splits data based on a feature
* Recursively partitions the dataset

Example:

```
Is income > 50K?
   ├── Yes → Predict High Income
   └── No  → Check Age
```

## 2. Splitting Criteria

The goal at each node: Find the best feature and threshold to split the data

### For Classification

Common impurity measures:

#### **Gini Impurity**

$$
G = 1 - \sum p_i^2
$$

#### **Entropy**

$$
H = -\sum p_i \log_2 p_i
$$

Lower impurity = better split

## 3. Information Gain

A split is evaluated by how much it reduces impurity:

$$
\text{Information Gain} = \text{Parent Impurity} - \text{Weighted Child Impurity}
$$

Choose split with **highest information gain**

## 4. Tree Structure

* **Root node** → top of tree
* **Internal nodes** → decisions
* **Leaf nodes** → predictions

## 5. Training Decision Trees

Greedy algorithm:

1. Choose best split
2. Partition data
3. Repeat recursively

Limitation: Does not guarantee global optimal tree

## 6. Classification vs Regression Trees

### Classification Tree

* Predicts class label
* Uses Gini or entropy

### Regression Tree

* Predicts numeric value
* Minimizes variance (MSE)

## 7. Overfitting and Regularization

Decision trees can easily **overfit**.

### Symptoms:

* Very deep tree
* Perfect training accuracy
* Poor generalization

### Regularization Techniques

* `max_depth`
* `min_samples_split`
* `min_samples_leaf`
* `max_leaf_nodes`

These limit tree growth

## 8. Pruning

### Pre-pruning

* Stop tree early using hyperparameters

### Post-pruning

* Grow full tree → remove unnecessary branches


## 9. Decision Boundary

* Decision trees create **axis-aligned splits**
* Resulting boundaries are:

  * Rectangular
  * Not smooth

## 10. Limitations of Decision Trees

* High variance (unstable)
* Sensitive to small data changes
* Can overfit easily
* Poor extrapolation in regression

## 11. Implementation (Scikit-Learn)

```python
from sklearn.tree import DecisionTreeClassifier

tree = DecisionTreeClassifier(max_depth=3)
tree.fit(X_train, y_train)
```

## 12. Interpretability

Decision trees are:

* Easy to visualize
* Explainable
* Useful for teaching and debugging

# Glossary

| Term                | Definition                                  |
| ------------------- | ------------------------------------------- |
| Decision Tree       | Model that splits data using decision rules |
| Node                | A point where a decision is made            |
| Root Node           | Top of the tree                             |
| Leaf Node           | Final prediction node                       |
| Split               | Dividing data based on feature              |
| Gini Impurity       | Measure of class impurity                   |
| Entropy             | Measure of disorder                         |
| Information Gain    | Reduction in impurity after split           |
| Overfitting         | Model fits noise instead of pattern         |
| Pruning             | Reducing tree size to avoid overfitting     |
| max_depth           | Maximum depth of tree                       |
| min_samples_split   | Minimum samples to split a node             |
| Regression Tree     | Predicts continuous values                  |
| Classification Tree | Predicts discrete labels                    |

# Review Questions

### 1. What is the main idea behind a decision tree?

<details>
<summary>Answer</summary>

A decision tree recursively splits data using feature-based rules to make predictions.
</details>

### 2. Why do we need impurity measures?

<details>
<summary>Answer</summary>

They quantify how “mixed” a node is and help choose the best split.
</details>

### 3. What is the difference between Gini impurity and entropy?

<details>
<summary>Answer</summary>

* Both measure impurity
* Entropy uses logarithms (more sensitive)
* Gini is faster and commonly used
</details>

### 4. Why are decision trees prone to overfitting?

<details>
<summary>Answer</summary>

Because they can keep splitting until they perfectly fit the training data, capturing noise.
</details>

### 5. What is pruning?

<details>
<summary>Answer</summary>

Pruning removes unnecessary branches to improve generalization.
</details>

### 6. How do you choose the best split?

<details>
<summary>Answer</summary>

By maximizing information gain (or minimizing impurity).
</details>

### 7. What happens if a tree is too deep?

<details>
<summary>Answer</summary>

It overfits and performs poorly on new data.
</details>

### 8. Why are decision tree boundaries not smooth?

<details>
<summary>Answer</summary>

Because splits are axis-aligned (horizontal/vertical cuts).
</details>

### 9. How do hyperparameters help control overfitting?

<details>
<summary>Answer</summary>

They limit tree growth (depth, samples per node, etc.).
</details>

### 10. When would you use a regression tree instead of a classification tree?

<details>
<summary>Answer</summary>

When predicting continuous numeric values.
</details>

### 11. What does this do?

```python
DecisionTreeClassifier(max_depth=3)
```

<details>
<summary>Answer</summary>

Creates a decision tree limited to depth 3 to reduce overfitting.
</details>

### 12. What happens during `.fit()`?

<details>
<summary>Answer</summary>

The tree is built by recursively splitting the data based on impurity reduction.
</details>

### 13. Why might a small change in data drastically change a decision tree?

<details>
<summary>Answer</summary>

Because decision trees are **high variance models**:

* A small change can lead to a different split early in the tree
* This propagates and changes the entire structure
</details>
