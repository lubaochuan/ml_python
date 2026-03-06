# Chapter 5 Summary: Support Vector Machines (SVM)

## 1. What is a Support Vector Machine?

A **Support Vector Machine (SVM)** is a supervised machine learning algorithm used for **classification** and **regression** that finds the decision boundary separating classes with the **largest possible margin**.

The key idea is to choose the decision boundary that maximizes the distance between the boundary and the closest data points from each class.

These closest points are called **support vectors**.

## 2. Linear SVM Classification

* In the simplest case, the data is **linearly separable**.
* A **decision boundary (hyperplane)** separates the classes.
* For two features the hyperplane is a **line**:

## 3. The Margin

The **margin** is the distance between the decision boundary and the nearest training points.

SVM chooses the boundary that **maximizes the margin**.

Benefits of a large margin:

* better **generalization**
* more **robust to noise**
* less likely to overfit

## 4. Support Vectors

**Support vectors** are the training points that lie:

* on the margin boundary, or
* inside the margin (for soft-margin SVM)

Only the support vectors influence the decision boundary.

Points far from the boundary **do not affect the model**.

## 5. Hard Margin vs Soft Margin

### Hard Margin SVM

Assumes the data is **perfectly separable**.

Requirements:

* no misclassification allowed
* margin must not contain any points
* very sensitive to **outliers**

### Soft Margin SVM

Allows **some points to violate the margin**.

This is controlled by parameter **C**.

## 6. The C Hyperparameter

The parameter **C** controls the trade-off between:

* **large margin**
* **classification errors**

| C value | Behavior                            |
| ------- | ----------------------------------- |
| Small C | wider margin, more mistakes allowed |
| Large C | narrow margin, fewer mistakes       |

Small C → stronger **regularization**.

## 7. Feature Scaling

SVM is sensitive to feature scales.

Example:

* feature range 0–1
* feature range 0–1000

The larger scale feature dominates.

Therefore **feature scaling is essential**.

# 8. Nonlinear SVM Classification

Many datasets are **not linearly separable**.

SVM solves this using **kernels**.

A kernel allows SVM to find a **nonlinear boundary** without explicitly transforming the data.

## 9. Polynomial Kernel

Adds polynomial features implicitly.

Key parameters:

* **degree**
* **coef0**
* **C**

## 10. Gaussian RBF Kernel

The **Radial Basis Function (RBF)** kernel is the most widely used SVM kernel.

It measures similarity between points.

Parameter **gamma** controls the influence of each point.

| Gamma       | Behavior         |
| ----------- | ---------------- |
| Small gamma | smooth boundary  |
| Large gamma | complex boundary |


# Key Takeaways

Support Vector Machines:

* maximize the **margin**
* depend only on **support vectors**
* use **C** to control regularization
* handle nonlinear problems using **kernels**
* require **feature scaling**

# Review Questions

### 1. What is the goal of a Support Vector Machine?

<details>
<summary>answer</summary>

To find a decision boundary that maximizes the margin between classes.
</details>

### 2. What are support vectors?

<details>
<summary>answer</summary>

Training points closest to the decision boundary that determine the position of the boundary.
</details>

### 3. What is the margin?

<details>
<summary>answer</summary>

The distance between the decision boundary and the closest training points.
</details>

### 4. Why does SVM prefer a large margin?

<details>
<summary>answer</summary>

A larger margin improves generalization and reduces overfitting.
</details>

### 5. What does the hyperparameter C control?

<details>
<summary>answer</summary>

The trade-off between maximizing the margin and minimizing classification errors.
</details>

### 6. What happens when C is very large?

<details>
<summary>answer</summary>

The model tries to classify every training example correctly, leading to a smaller margin and potential overfitting.
</details>

### 7. Why is feature scaling important for SVM?

<details>
<summary>answer</summary>

Because SVM relies on distances between points, and features with larger scales can dominate the model.
</details>

### 8. What problem do kernels solve?

<details>
<summary>answer</summary>

They allow SVM to create nonlinear decision boundaries.
</details>

### 9. What does the gamma parameter control in the RBF kernel?

<details>
<summary>answer</summary>

How far the influence of a single training point extends.
</details>


# SVM Hyperparameter Concept Map

```
                     Support Vector Machine
                              |
              ------------------------------------
              |                                  |
         Linear SVM                         Kernel SVM
              |                                  |
           Parameter                           Kernel
              C                             /      |      \
                                          Linear  Poly    RBF
                                                   |       |
                                                 degree     gamma
                                                  coef0
```
