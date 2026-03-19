# Videos

## StatQuest: Linear Regression, Clearly Explained!!!
https://www.google.com/search?q=https://youtu.be/7ArmBVF2dCs

This video provides a clear and intuitive breakdown of **Linear Regression**, focusing on how we fit a line to data and how we evaluate the quality of that fit.

### **Key Concepts Covered:**

* **Fitting a Line with Least Squares**
The core method for finding the "best" line is to minimize the **Sum of Squared Residuals**.
* **Residuals** are the distances between the actual data points and the line.
* By rotating the line and calculating the sum of these squared distances, we find the specific rotation where the error is at its absolute minimum.


* **Quantifying the Relationship with $R^{2}$:**
 tells us what percentage of the variation in the data (e.g., Mouse Size) can be explained by the predictor (e.g., Mouse Weight).
* It compares the variation around the **Mean** (the simplest possible model) to the variation around the **Fitted Line**.
* **Example** An $R^{2}$ of 0.6 means that 60% of the variation is explained by the model, representing a 60% reduction in variance.


* **Statistical Significance and the F-Statistic:**
A high $R^{2}$ isn't enough; we need to know if the relationship is reliable or just due to random chance. This is determined using the **F-statistic**.
* **F-Statistic Formula** It calculates the ratio of the variation explained by the model to the variation that remains unexplained.
* **Degrees of Freedom** The calculation takes into account the number of parameters used and the sample size (`n`).


* **Calculating the P-Value**
The F-statistic is used to derive a **p-value**. A small p-value (typically < 0.05) indicates that the relationship we've found is statistically significant and unlikely to have occurred by pure coincidence.

## R-squared, Clearly Explained!!!
https://www.google.com/search?q=https://youtu.be/2AQKmw14mHM

This video explains **** (R-squared), a metric used to quantify how well a model explains the variation in a dataset. While it is related to the correlation coefficient ($r$), $R^{2}$ is often preferred because its interpretation as a percentage is more intuitive.

### **Key Concepts Covered:**

* **What is $R^{2}$?**
 is a metric of correlation that measures the proportion of variance in the dependent variable ($Y$) that is predictable from the independent variable ($X$). Unlike plain old $r$, which can be hard to compare (is $r=0.7$ twice as good as $r=0.5$?), $R^{2}$ provides a direct linear scale of "goodness of fit."
* **How to Calculate**
The calculation compares two types of variation:
1. **Variation around the Mean:** The sum of squared differences between the data points and the average (the "baseline" or simplest model).
2. **Variation around the Fitted Line:** The sum of squared differences between the data points and the regression line.

  The formula is: $$R^2 = \frac{\text{Var(mean)} - \text{Var(line)}}{\text{Var(mean)}}$$

* **Intuitive Interpretation**
* **Example 1 (Strong Relationship):** An $R^2$ of **0.81** means that **81%** of the variation in the data is explained by the relationship between the variables (e.g., Mouse Size vs. Weight). There is 81% less variation around the line than there is around the mean.
* **Example 2 (Weak Relationship):** An $R^2$ of **0.06** means the relationship only accounts for **6%** of the variation, implying that 94% of the variation is due to other unknown factors.

* $R$ vs. $R^2$
  * $R^2$ is literally the square of the correlation coefficient $r$.
  * **The Benefit:** If $r=0.7$, then $R^2=0.49$ (approx 50% explained). If $r=0.5$, then $R^2=0.25$ (25% explained). This makes it easy to see that $r=0.7$ is actually twice as good at explaining variation as $r=0.5$.
  * **The Limitation:** $R^2$ does not indicate the direction (positive or negative) of the relationship because squared numbers are never negative.

