# Chapter 3 Summary

## Classification

### Big Picture

Chapter 3 introduces **classification problems**, where the goal is to assign inputs to **discrete categories** rather than predicting continuous values. Using the MNIST digit dataset as a running example, the chapter emphasizes that **choosing the right evaluation metric** is often more important than choosing the model itself—especially when classes are imbalanced or errors have unequal costs.

## 1. What Is Classification?

Classification tasks involve predicting:

* Binary labels (e.g., spam vs. not spam)
* Multi-class labels (e.g., digits 0–9)
* Multi-label outputs (multiple labels per instance)

The chapter begins with a **binary classification task** (“Is this digit a 5?”) to simplify reasoning before extending to more complex cases.

## 2. Training a Binary Classifier

A simple linear classifier (e.g., **SGDClassifier**) is trained on MNIST.

Key ideas:

* Large datasets often require **stochastic or mini-batch methods**
* Linear classifiers are fast and scalable
* Training accuracy alone is misleading

## 3. Performance Measures

This is the conceptual core of the chapter.

### Confusion Matrix

A table summarizing prediction outcomes:

* True Positives
* False Positives
* True Negatives
* False Negatives

From this matrix, many metrics are derived.

### Precision and Recall

* **Precision**: How many predicted positives are actually correct? When the model predicts positive, how often is it correct?
$$
\text{Precision} = \frac{TP}{TP + FP}
$$
* **Recall**: How many **actual** positives were correctly identified? (TPR)
$$
\text{Recall} = \frac{TP}{TP + FN}
$$

* **Accuracy**: Out of all predictions, how many were correct?
  * Correct predictions: TP + TN
  * Total: TP + TN + FP + FN
  * Accuracy = (TP + TN) / Total

The chapter highlights the **precision–recall tradeoff** and why different applications prioritize different metrics (e.g., medical screening vs. spam filtering).

### F1 Score

The harmonic mean of precision and recall, useful when a balance is needed.

## 4. Decision Thresholds

Many classifiers output **scores**, not labels.

* The **decision threshold** determines how scores map to class labels.
* Adjusting the threshold changes precision and recall.
* This enables models to be tuned for business or ethical priorities.

## 5. ROC Curves and AUC

Another evaluation framework:

* **ROC curve** plots True Positive Rate vs. False Positive Rate.
* **AUC** summarizes performance across all thresholds.

The chapter explains:

* ROC AUC is useful for comparing classifiers
* Precision–Recall (PR) curves are often better for **imbalanced datasets**

## 6. Multiclass Classification

Two common strategies:

* **One-vs-Rest (OvR)**
* **One-vs-One (OvO)**

Many Scikit-Learn classifiers support multiclass classification natively or through wrappers.

## 7. Error Analysis

Beyond metrics:

* Inspect misclassified examples
* Look for systematic patterns
* Understand *why* the model fails

This step informs better feature engineering and data collection.

## 8. Multilabel Classification

In multilabel tasks:

* Each instance can belong to multiple classes
* Evaluation requires specialized metrics (e.g., Hamming loss)

## 9. Multioutput Classification

An extension where:

* Each instance has multiple outputs
* Each output may be multiclass

This is useful for structured prediction problems.

## Key Takeaways

* Accuracy alone is often **misleading**.
* Evaluation metrics encode **values and priorities**.
* Threshold tuning is a powerful, underused tool.
* Error analysis is essential for improvement and trust.
* Classification performance must be interpreted in context.

# Glossary of Key Terms

### Classification

A supervised learning task where outputs are discrete labels.

### Binary Classification

Classification with **two** possible classes.

### Multiclass Classification

Classification with **more than two** classes.

### Multilabel Classification

Each instance can belong to multiple classes **simultaneously**.

### Multioutput Classification

Each instance has multiple target variables.

### Confusion Matrix

A table summarizing correct and incorrect predictions.

### True Positive (TP)

A positive instance correctly classified.

### False Positive (FP)

A negative instance incorrectly classified as positive.

### True Negative (TN)

A negative instance correctly classified.

### False Negative (FN)

A positive instance incorrectly classified as negative.

### Precision

The fraction of positive predictions that are correct.

### Recall (Sensitivity)

The fraction of actual positives that are correctly identified.

### F1 Score

The harmonic mean of precision and recall.

### Decision Threshold

A cutoff value used to convert scores into class labels.

### Score

A continuous output from a classifier indicating confidence.

### ROC Curve

A plot of True Positive Rate vs. False Positive Rate.

### AUC (Area Under the Curve)

A scalar summary of the ROC curve.

### Precision–Recall Curve

A plot showing the tradeoff between precision and recall.

### Class Imbalance

A situation where some classes appear much more frequently than others.

### One-vs-Rest (OvR)

A strategy for multiclass classification using binary classifiers.

### One-vs-One (OvO)

A strategy that trains classifiers for every pair of classes.

### Error Analysis

The process of examining misclassifications to improve models.
