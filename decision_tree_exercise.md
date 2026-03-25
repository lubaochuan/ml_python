# Step-by-Step Gini Calculations Worksheet

Learn how a decision tree chooses splits by **computing Gini impurity manually**.

## Gini Formula

$$
Gini = 1 - \sum_{k=1}^{K} p_k^2
$$

## Dataset

| Hours | Passed |
| ----- | ------ |
| 2     | No     |
| 3     | No     |
| 4     | Yes    |
| 5     | Yes    |
| 6     | Yes    |

## Part 1: Compute Root Gini

Step 1: Count classes

* Yes = ___
* No = ___

Step 2: Compute probabilities

* $p_{yes}$ = ___
* $p_{no}$ = ___

Step 3: Compute Gini

$$
Gini = 1 - (p_{yes}^2 + p_{no}^2)
$$

Final Answer: **Gini(root) = ______**

## Part 2: Try Split (Hours ≤ 3)

### Left Node (Hours ≤ 3)

* Data points: ______
* Yes = ___
* No = ___

$Gini_{left}$ = ______

### Right Node (Hours > 3)

* Data points: ______
* Yes = ___
* No = ___

$Gini_{right}$ = ______

## Part 3: Weighted Gini

$$
Gini_{split} = \frac{n_{left}}{n} Gini_{left} + \frac{n_{right}}{n} Gini_{right}
$$

Fill in:

* $n_{left}$ = ___
* $n_{right}$ = ___
* n = ___

Final Answer:
**Gini(split) = ______**

## Part 4: Compare

Which is better?

* Root Gini = ______
* Split Gini = ______

Does this split improve purity? Why?

## Part 5 (Challenge)

Try another split:

**Hours ≤ 4**

Repeat the same steps and compare results.
