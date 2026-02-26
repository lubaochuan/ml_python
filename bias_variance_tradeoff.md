When building a classification model, you’re trying to balance two types of mistakes in how the model *learns*, not just how it predicts.

### Bias = Being Too Simple

Bias happens when the model is **too simple** to capture the real patterns in the data.

* It makes strong assumptions.
* It misses important structure.
* It performs poorly on both training and test data.

Think of bias like using a straight line to separate data that clearly needs a curve.

The model **underfits**.

### Variance = Being Too Sensitive

Variance happens when the model is **too complex** and reacts too much to small details or noise in the training data.

* It memorizes the training data.
* It performs very well on training data.
* It performs poorly on new (test) data.

Think of variance like drawing a super wiggly boundary that perfectly separates training points but wouldn’t work for new data.

The model **overfits**.

### The Tradeoff

* If you make the model **simpler**, bias increases but variance decreases.
* If you make the model **more complex**, bias decreases but variance increases.

You can’t usually minimize both at the same time — improving one often worsens the other.

### The Goal

Find the “sweet spot”:

* Complex enough to capture real patterns.
* Simple enough to generalize to new data.

That balance is the **bias–variance tradeoff**.
