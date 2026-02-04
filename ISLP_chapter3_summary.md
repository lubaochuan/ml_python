# Chapter 3 Summary: Linear Regression

## 1. What Is Linear Regression?

**Linear regression** is one of the simplest and most important tools in machine learning and statistics.

Its goal is to **predict a number** (a *quantity*) using one or more inputs.

Example questions:

* How does house price depend on size?
* How does exam score depend on study hours?
* How does sales depend on advertising spend?

The basic idea:

> Draw the **best-fitting straight line (or plane)** through data.

Even when the real world isn’t perfectly linear, linear regression often works surprisingly well.

## 2. Simple Linear Regression (One Input)

### The Model (Conceptually)

We try to predict an output **Y** using a single input **X**.

* **Intercept**: where the line starts when X = 0
* **Slope**: how much Y changes when X increases by 1

Intuition:

* Slope = “effect size”
* Intercept = “baseline value”

## 3. What Does “Best Fit” Mean?

The model chooses the line that:

* Minimizes the **average squared vertical distance** between the data points and the line

This is called **least squares**.

Why squares?

* Penalizes big mistakes more
* Easier to compute and optimize

Key idea:

> We are minimizing **error**, not finding a perfect fit.

## 4. Making Predictions

Once the line is learned:

* Plug in a new X
* Get a predicted Y

Important:

* Predictions **outside the data range** (extrapolation) are risky
* Linear regression is best at **interpolation**

## 5. Interpreting the Coefficients

Linear regression is popular because it is **interpretable**.

* **Slope** answers:

  > “If X increases by 1, how much does Y change?”
* **Sign of slope**:

  * Positive → Y increases with X
  * Negative → Y decreases with X

## 6. Measuring Model Quality

### R² (R-squared)

* Measures **how much of the variation in Y** is explained by the model
* Ranges from 0 to 1

Interpretation:

* R² = 0 → model explains nothing
* R² = 1 → model explains everything

High R² does **not** guarantee a good model.

### Residuals

* **Residual = actual value − predicted value**
* Residual plots help diagnose problems

Good signs:

* Random scatter around zero

Bad signs:

* Patterns or curves → model is missing something

## 7. Multiple Linear Regression (Many Inputs)

Now we use **multiple inputs**:

* Size
* Location
* Age
* Number of rooms

Each input has its own coefficient.

Interpretation:

> Each coefficient describes the effect of that variable **holding all others constant**.

This is powerful—but easy to misunderstand.

## 8. Categorical Variables (Qualitative Inputs)

Linear regression can handle categories using **dummy variables**.

Example:

* Neighborhood = A, B, C

We convert categories into:

* 0s and 1s

Key idea:

> Categories are turned into numbers **without implying order**.

## 9. Interaction Terms

Sometimes variables **affect each other**.

Example:

* Advertising on TV works better when online ads are also used

An **interaction term** lets the model capture this combined effect.

Intuition:

> “The effect of X depends on the value of Z.”

## 10. Assumptions (Conceptual, Not Mathematical)

Linear regression assumes:

1. Relationship is roughly linear
2. Errors are independent
3. Errors have constant spread
4. No extreme outliers dominate

Violating assumptions doesn’t always break the model—but it affects reliability.

## 11. When Linear Regression Works Well

* Relationship is roughly linear
* Interpretability matters
* Data size is moderate
* Noise is not extreme

## 12. When It Struggles

* Strongly nonlinear relationships
* Complex interactions
* High overfitting risk without care

## Big Takeaways

* Linear regression is **simple, powerful, and interpretable**
* Coefficients have meaningful real-world interpretations
* Model evaluation matters more than fitting the line
* Always check assumptions and residuals

# Review Questions

### 1. Why is linear regression still widely used despite being simple?

<details>
<summary>Example Answer</summary>

Because it is easy to interpret, fast to compute, and often performs well even when assumptions are imperfect.
</details>

### 2. What does the slope of a linear regression line represent in plain language?

<details>
<summary>Example Answer</summary>

It represents the expected change in the output when the input increases by one unit.
</details>

### 3. Why do we minimize squared errors instead of absolute errors?

<details>
<summary>Example Answer</summary>

They penalize large mistakes more strongly and lead to efficient mathematical solutions.
</details>

### 4. What does R² tell us—and what does it *not* tell us?

<details>
<summary>Example Answer</summary>

R² measures explained variation but does not guarantee **causality** or good predictions.
</details>

### 5. Why is extrapolation risky in linear regression?

<details>
<summary>Example Answer</summary>

Because the model has no evidence that the linear relationship holds outside the observed data range.
</details>

### 6. How does multiple linear regression differ from simple linear regression?

<details>
<summary>Example Answer</summary>

Multiple regression uses several inputs and estimates the effect of each one separately.
</details>

### 7. What does it mean to interpret a coefficient “holding other variables constant”?

<details>
<summary>Example Answer</summary>

It means isolating the effect of one variable while accounting for the influence of others.
</details>

### 8. Why do we need dummy variables for categorical data?

<details>
<summary>Example Answer</summary>

They allow categorical data to be included numerically without implying an order.
</details>

### 9. What is an interaction term, and when might it be useful?

<details>
<summary>Example Answer</summary>

They model situations where the effect of one variable depends on another.
</details>

### 10. What kinds of patterns in residual plots suggest problems?

<details>
<summary>Example Answer</summary>

Curves, trends, or changing spread suggest nonlinearity or missing variables.
</details>