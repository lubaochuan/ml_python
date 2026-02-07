# Chapter 4 Summary: Classification

## 1. What Is Classification?

**Classification** is about predicting a **category (label)** rather than a number.

Examples:

* Spam vs not spam
* Fraud vs not fraud
* Disease vs no disease
* Pass vs fail

Conceptually:

> The model learns a rule that maps inputs → **class labels**.

Unlike regression (Chapter 3), the output is **discrete**, not continuous.

## 2. From Scores to Decisions

Most classifiers internally compute a **score** (or probability) and then apply a **decision rule**.

Example:

* If probability(spam) ≥ 0.5 → spam
* Else → not spam

Key idea:

> Classification is often about **ranking likelihood**, then choosing a cutoff.

## 3. Logistic Regression (Core Classifier)

Despite the name, **logistic regression is a classification method**.

What it does:

* Combines inputs linearly (like linear regression)
* Passes the result through a **sigmoid (S-shaped) function**
* Outputs a **probability between 0 and 1**

Why it’s popular:

* Interpretable
* Stable
* Outputs probabilities (very useful in practice)

Intuition:

> Logistic regression draws a **linear decision boundary**, but predicts probabilities, not just labels.

## 4. Probabilities vs Labels

A key advantage of logistic regression:

* It tells you **how confident** it is

Example:

* 0.51 → barely positive
* 0.99 → very confident

This matters when:

* Costs of mistakes are unequal
* You want human review for uncertain cases

## 5. Decision Boundaries

A **decision boundary** separates classes in feature space.

* Linear models → straight lines (or planes)
* Flexible models (like k-NN) → curved, complex boundaries

Tradeoff:

* Simple boundary → easier to interpret, may underfit
* Complex boundary → flexible, may overfit

## 6. k-Nearest Neighbors (k-NN) for Classification

k-NN classifies a point by:

1. Finding its **k nearest neighbors**
2. Taking a **majority vote**

Behavior:

* Small k → very flexible, high variance
* Large k → smoother, higher bias

Mental model:

> “Tell me who your neighbors are, and I’ll tell you who you are.”

## 7. Comparing Logistic Regression and k-NN

| Aspect           | Logistic Regression | k-NN           |
| ---------------- | ------------------- | -------------- |
| Type             | Parametric          | Non-parametric |
| Boundary         | Linear              | Flexible       |
| Interpretability | High                | Low            |
| Data needed      | Less                | More           |
| Probabilities    | Yes                 | Not naturally  |

There is **no universally best classifier**—context matters.

## 8. Evaluating Classifiers: Accuracy Isn’t Enough

### Accuracy

* Fraction of correct predictions
* Can be misleading with **imbalanced classes**

Example:

* 99% “not fraud” → 99% accuracy, but useless

### Confusion Matrix

Shows:

* True positives
* True negatives
* False positives
* False negatives

Helps you understand **what kinds of mistakes** the model makes.

### Precision and Recall (Conceptually)

* **Precision**: When the model says “positive,” how often is it right?
* **Recall**: Of all real positives, how many did it find?

Different applications prioritize different metrics.

## 9. Threshold Choice Matters

Changing the cutoff (e.g., 0.5 → 0.7):

* Increases precision
* Decreases recall

Key idea:

> Classification performance depends on **how cautious or aggressive** you want the model to be.

## 10. When to Use Which Classifier

Use **logistic regression** when:

* Interpretability matters
* Data is limited
* A linear boundary is reasonable

Use **k-NN** when:

* You have lots of data
* The boundary is complex
* Interpretability is less important

## Big Takeaways from Chapter 4

* Classification predicts **categories**, not numbers
* Probabilities are often more useful than hard labels
* Accuracy alone can be misleading
* Decision boundaries reflect model flexibility
* Model choice depends on data, goals, and costs of errors

# Review Questions

### 1. How is classification different from regression?

<details>
<summary>Example Answer</summary>

Classification predicts categories; regression predicts numeric values.
</details>


### 2. Why is logistic regression considered a classification method even though it has “regression” in its name?

<details>
<summary>Example Answer</summary>

It models probabilities for class membership and uses those probabilities to classify, despite using a regression-like formula internally.
</details>

### 3. What is a decision boundary, and why does its shape matter?

<details>
<summary>Example Answer</summary>

It separates regions of different predicted classes; shape reflects model flexibility and bias–variance tradeoff.
</details>

### 4. Why can accuracy be a misleading metric?

<details>
<summary>Example Answer</summary>

Accuracy hides the types of errors and fails under class imbalance.
</details>

### 5. What information does a confusion matrix provide that accuracy does not?

<details>
<summary>Example Answer</summary>

It shows false positives and false negatives, revealing error structure.
</details>

### 6. In what situations are probabilities more useful than class labels?

<details>
<summary>Example Answer</summary>

When decisions have different costs, or when uncertain cases need human review.
</details>

### 7. How does the choice of k affect k-NN classification?

<details>
<summary>Example Answer</summary>

Small k → high variance; large k → higher bias and smoother boundaries.
</details>

### 8. Why might logistic regression outperform k-NN on small datasets?

<details>
<summary>Example Answer</summary>

It has fewer parameters and is more stable than highly flexible methods.
</details>

### 9. What happens when you raise the classification threshold?

<details>
<summary>Example Answer</summary>

Increases precision but reduces recall.
</details>

### 10. Why is there no single “best” classification algorithm?

<details>
<summary>Example Answer</summary>

Performance depends on data size, noise, class balance, interpretability needs, and error costs.
</details>
