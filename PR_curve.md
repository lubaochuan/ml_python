## PR-curve Intuition

### Foundations

Precision answers what question?

A) Of all actual positives, how many did we catch?
B) Of all predicted positives, how many were correct?
C) Of all negatives, how many did we reject?

answer B

Recall answers what question?

A) Of all predicted positives, how many were correct?
B) Of all actual positives, how many did we catch?
C) Of all actual negatives, how many did we reject?

answer B

### Build from Numbers

Suppose:

* 1000 total samples
* 50 positives
* 950 negatives

A model at some threshold produces:

TP = 40, FP = 60, FN = 10, TN = 890

Compute:
* Precision = ?
* Recall = ?

answer
precision = TP/(TP+FP) = 40/(40+60)=2/5
recall = TP/(TP+FN)=40/(40+10)=4/5

Is the model trustworthy when it predicts positive?

answer no, when it predicts positive, it is wrong 60% of the time.

# Threshold Movement Intuition

If we LOWER the threshold (make it easier to predict positive):

* What happens to recall?
* What happens to precision?

Explain *why*, not just the direction.

answer
When you lower the threshold:

You label more samples as positive

So TP increases (good)

But FP also increases (bad)
As TP increases, recall increases.

When you lower the threshold:
TP increases
FP increases
But usually FP increases faster than TP
So precision drops.

### Extreme Cases

### Case A: Predict EVERYTHING positive recall = 1 precision = positive cases/total

What are:

* Recall?
* Precision?

(Hint: precision depends on class prevalence.)

answer: recall = 1 precision = (# of positives) / (total samples)

The baseline of the PR curve equals the class prevalence.

### Case B: Predict NOTHING positive

What are:

* Recall?
* Precision?

answer: recall (FPR) = 0, precision = undefined

### Geometry of PR Curve

Now think conceptually.

The PR curve plots:

* X-axis = Recall
* Y-axis = Precision

When threshold decreases, which direction do we move along the PR curve?

Left to right?
Right to left?
Upward?
Downward?

Explain the direction.

---

# 🔥 Part 6 — Imbalance Mastery

Suppose only 1% of data is positive.

If you randomly guess positive with probability 1%, what is your expected precision?

(Hint: it equals the prevalence.)

---

# 🔥 Part 7 — Deep Insight Question

Why is the baseline of a PR curve equal to the class prevalence, but the ROC baseline is 0.5?

This question separates surface understanding from deep understanding.

---

Reply with your answers step by step.
I will correct and push your intuition deeper.

Let’s master PR curves properly.
