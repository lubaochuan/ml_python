
## Confusion Matrix

A medical test is used to detect a disease.

|          | Disease (+) | Disease (–) |
| -------- | ----------- | ----------- |
| Test (+) | 40          | 10          |
| Test (–) | 20          | 130         |

### Compute:
* TP, FP, FN, and TN
* **Sensitivity (Recall, TPR)**
* **Specificity (TNR)**

### Interpretation

* What does the sensitivity value mean?
* What does the specificity value mean?
* Which mistake is this test making more often — false positives or false negatives?

### Shift the Threshold

Now suppose we make the test **more strict** (harder to predict positive).

The new confusion matrix becomes:

|          | Disease (+) | Disease (–) |
| -------- | ----------- | ----------- |
| Test (+) | 30          | 3           |
| Test (–) | 30          | 137         |

Without calculating yet:
* Did sensitivity go up or down?
* Did specificity go up or down?
* Why?

Now compute them to confirm your intuition.

### Tradeoff

If we lower the classification threshold (making it easier to predict positive):

* Sensitivity ↑ or ↓ ?
* Specificity ↑ or ↓ ?

Why does this always happen?

<details>
<summary>answer</summary>

When you lower the threshold:

* You predict **positive more easily**
* More people are labeled positive

That means:

### What increases?

* TP increases (you catch more sick people)
* FP increases (you wrongly flag more healthy people)

So:

$$
\text{Sensitivity} = \frac{TP}{TP+FN} \uparrow
$$

$$
\text{Specificity} = \frac{TN}{TN+FP} \downarrow
$$

### If I make the threshold extremely low (almost always predict positive):

* Sensitivity → ?
* Specificity → ?

### If I make the threshold extremely high (almost never predict positive):

* Sensitivity → ?
* Specificity → ?

Answer:

Model predicts **positive for almost everyone**

What happens?

Everyone is labeled positive.

So:

* All actual positives are caught → **Sensitivity = 1**
* No actual negatives are correctly identified → **Specificity = 0**

Because:

* TN = 0
* FP = all negatives

Model predicts **negative for almost everyone**

What happens?

Everyone is labeled negative.

So:

* No actual positives are caught → **Sensitivity = 0**
* All actual negatives are correctly identified → **Specificity = 1**
</details>

Sensitivity and specificity are like a seesaw:

```
High Sensitivity  ←──── threshold ────→  High Specificity
```

You can slide along this tradeoff, but you cannot maximize both simultaneously (unless the classifier is perfect).


#### ROC Intuition

Quick conceptual check:

The ROC curve plots:

* X-axis = False Positive Rate (1 − Specificity)
* Y-axis = True Positive Rate (Sensitivity)

Now answer:

1. Where is the “always positive” classifier located in ROC space?
2. Where is the “always negative” classifier located?
3. Where is a perfect classifier located?

Think geometrically.

<details>
<summary>answer</summary>

ROC curve axes:

* **X-axis** = False Positive Rate (FPR) = 1 − Specificity
* **Y-axis** = True Positive Rate (TPR) = Sensitivity

So:

* Bottom-left = (0,0)
* Top-right = (1,1)

Always Predict NEGATIVE
* You never predict positive.
* So TP = 0 → TPR = 0
* FP = 0 → FPR = 0

So the point is:

$$
(0,0)
$$

Bottom-left corner.

This classifier detects nothing.

Always Predict POSITIVE

What happens?
* You label everyone positive.
* So TPR = 1 (you catch all positives)
* FPR = 1 (you falsely label all negatives positive)

So the point is:

$$
(1,1)
$$

Top-right corner.

This classifier predicts everything as positive.

Perfect Classifier

What happens?
* TPR = 1 (catch all positives)
* FPR = 0 (no false positives)

So the point is:

$$
(0,1)
$$

Top-left corner.

This is the holy grail.
</details>

### Why ROC Curves Bend

As you lower the threshold:
* You move from (0,0)
* Up toward (0,1)
* Then eventually toward (1,1)

The curve traces how TPR increases as FPR increases.

Good models:

* Hug the **top-left corner**
* Have large **AUC**

Random guessing:

* Lies along the diagonal from (0,0) to (1,1)

Because:

* TPR ≈ FPR

Why is a model below the diagonal worse than random?

### Core Intuition

* Above diagonal → useful signal
* On diagonal → no signal
* Below diagonal → inverted signal
* Top-left → perfect
* Bottom-left → useless (never predicts positive)
* Top-right → useless (always predicts positive)

Why is AUC threshold-independent, but sensitivity and specificity are threshold-dependent?

<details>
<summary>answer</summary>

For a fixed threshold:

* You make one set of predictions.
* That gives one confusion matrix.
* That produces one sensitivity and one specificity.

If you change the threshold → predictions change → confusion matrix changes → sensitivity and specificity change.

So they are tied to **one operating point**.

AUC does not commit to a single threshold.

Instead:

* It considers **all possible thresholds**
* It measures how well the model ranks positives above negatives overall

Mathematically:

> AUC = Probability(model ranks a random positive higher than a random negative)

So AUC measures the **ranking quality**, not the decision rule.
</details>