* Multiclass Classification
  * Every instance belongs to exactly one category. Even if there are 100 possible categories, the model must pick just one.
  * Examples: Identifying a digit (it's either a 4 or a 5, not both), or assigning a news article to a single "Politics" or "Sports" section. A radio button group with each button for a category.

* Multilabel Classification
  * Every instance can belong to zero, one, or multiple categories simultaneously. The labels are **not** mutually exclusive. You are predicting multiple binary properties. Each label is essentially a "Yes/No" or "0/1".
  * Example: A photo can be tagged as "beach," "sunset," AND "vacation" all at the same time. Does the photo have a beach? Yes. Does it have a dog? No. A group of "checkboxes", each of which can be chosen independently.

* Multioutput (or Multi-target) Classification
  * You are predicting multiple categorical properties, where each property can have **more than two** possible values.
  * Example: You predict [Type, Color, Size]: Type could be (Shirt, Pants, Hat); Color could be (Red, Blue, Green, Yellow); Size could be (S, M, L, XL). Each of those "outputs" is its own **multiclass** problem being solved simultaneously by one model.

* Precision
  * It answers the question: *"Of all the instances the model predicted as positive, how many were actually positive?"*
  $$\text{Precision} = \frac{\text{True Positives (TP)}}{\text{True Positives (TP)} + \text{False Positives (FP)}}$$
  * Think of **Precision** as **"Pre-decision"** caution. If you have high precision, you are very careful about calling something "Positive" because you don't want to be wrong (you want to avoid False Positives).

| Formula | Metric Name | What it tells you |
| --- | --- | --- |
| **TP / (TP + FP)** | **Precision** | How "trustworthy" the model is when it says "Positive." |
| **TP / (TP + FN)** | **Recall** | How "thorough" the model is at finding all positives. |
| **TN / (TN + FP)** | **Specificity** | How good the model is at identifying the negatives. |
| **(TP+TN) / Total** | **Accuracy** | The overall percentage of correct guesses. |

* PR Curve
  * It plots Precision vs. Recall.

* ROC Curve
  * It plots True Positive Rate vs. **False Positive Rate**.

| Feature | ROC Curve | PR Curve |
| --- | --- | --- |
| **Best Use Case** | Balanced classes / general performance. | Imbalanced classes (rare positive class). |
| **Includes "True Negatives"?** | Yes (via False Positive Rate). | No. |
| **Sensitivity to Imbalance** | Low (can look "too good" on bad models). | High (reveals if the model struggles with the minority). |

* Recall
  * (also known as Sensitivity) It measures the ability of a model to find all the relevant cases within a dataset. Mathematically, it is defined as:
$$\text{Recall} = \frac{\text{True Positives}}{\text{True Positives} + \text{False Negatives}}$$

* Specificity
  * (also known as the True Negative Rate) measures the ability of a model to correctly identify those who do not have a specific condition or characteristic. It answers the question: "Of all the people who are actually negative, how many did the model correctly identify as negative?"
  * The formula for Specificity is:
  $$\text{Specificity} = \frac{\text{True Negatives (TN)}}{\text{True Negatives (TN)} + \text{False Positives (FP)}}$$
  * While Recall is about not missing the "sick" people, Specificity is about not accidentally labeling "healthy" people as sick. A model with high specificity is very "skeptical"—it rarely cries wolf.
  * Imagine a test for a non-life-threatening condition where the follow-up treatment is very painful or expensive (like an invasive surgery). High Specificity: The test only comes back positive if it is absolutely certain you have the condition. This ensures that healthy people aren't subjected to unnecessary, painful surgery.
  * In email filtering, a False Positive is a "disaster"—it’s when an important work email from your boss gets sent to the Spam folder. High Specificity: The spam filter is designed to be highly specific so that it only marks an email as spam if it is definitely junk. It would rather let a few spam emails into your inbox (lower recall) than accidentally hide a single important message.