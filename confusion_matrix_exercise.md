This worksheet is designed for students to manually compute confusion matrices and metrics by manipulating numbers. It progresses from simple counting to interpreting trade-offs.

# Understanding Confusion Matrix, Precision & Recall (By Hand)

**Name:** ____________________

**Date:** ____________________

# Part 1 — Building a Confusion Matrix

### Scenario: Spam Detection

A spam classifier was tested on **100 emails**.

* 40 emails were actually spam.
* 60 emails were not spam.
* The model predicted 50 emails as spam.
* Of those 50 predicted spam emails:

  * 35 were actually spam.
  * 15 were not spam.

## Step 1: Fill in the Confusion Matrix

|                     | **Predicted Spam** | **Predicted Not Spam** | **Total** |
| ------------------- | ------------------ | ---------------------- | --------- |
| **Actual Spam**     |                    |                        |           |
| **Actual Not Spam** |                    |                        |           |
| **Total**           |                    |                        | 100       |

### Questions

1. What is the number of **True Positives (TP)**? _______

2. What is the number of **False Positives (FP)**? _______

3. What is the number of **False Negatives (FN)**? _______

4. What is the number of **True Negatives (TN)**? _______

# Part 2 — Compute the Metrics Manually

### Formulas

$$
\text{Precision} = \frac{TP}{TP + FP}
$$

$$
\text{Recall} = \frac{TP}{TP + FN}
$$

### Compute

1. Precision = ______________________

2. Recall = ______________________

3. Accuracy = ______________________

### Concept Questions

1. When the model says “spam,” how often is it correct?
2. Did the model miss many spam emails?
3. Would you prefer higher precision or higher recall in spam detection? Why?

# Part 3 — A Medical Screening Example

### Scenario: Cancer Screening Test

Out of 1,000 patients:

* 50 actually have cancer.
* The test predicts 120 patients as positive.
* Of those 120:

  * 45 actually have cancer.
  * 75 do not.

## Step 1: Fill in the Confusion Matrix

|                      | **Predicted Cancer** | **Predicted No Cancer** | **Total** |
| -------------------- | -------------------- | ----------------------- | --------- |
| **Actual Cancer**    |                      |                         |           |
| **Actual No Cancer** |                      |                         |           |
| **Total**            |                      |                         | 1000      |

## Step 2: Compute

1. Precision = ______________________
2. Recall = ______________________
3. Accuracy = ______________________

## Discussion Questions

1. Is the precision high or low?
2. Is the recall high or low?
3. Which error is more dangerous in this scenario?
4. Why might a doctor care more about recall than precision?

# Part 4 — Manipulating Numbers to See Trade-offs

Now imagine the screening test is made **more strict**.

It now predicts only 60 people as positive:

* 40 actually have cancer.
* 20 do not.

### Fill in the new confusion matrix:

|                      | **Predicted Cancer** | **Predicted No Cancer** |
| -------------------- | -------------------- | ----------------------- |
| **Actual Cancer**    |                      |                         |
| **Actual No Cancer** |                      |                         |

### Compute the new:

1. Precision = ______________________
2. Recall = ______________________

### Compare

1. Did precision increase or decrease?
2. Did recall increase or decrease?
3. Explain in one sentence what trade-off occurred.

# Final Reflection

1. What does **precision** measure?
2. What does **recall** measure?
3. Why is accuracy sometimes misleading?
4. Give one real-world example where recall is more important.
5. Give one real-world example where precision is more important.

