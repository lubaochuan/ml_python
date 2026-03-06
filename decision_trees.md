## Decision and Classification Trees, Clearly Explained!!!

http://www.youtube.com/watch?v=_L39rN6gz7Y

This **StatQuest** video provides a step-by-step explanation of how **Decision Trees**—specifically **Classification Trees**—work and how they are built from raw data.

**Tree Structure & Terminology:**
* **Root Node:** The very top of the tree where the first decision is made [[02:46](http://www.youtube.com/watch?v=_L39rN6gz7Y&t=166)].
* **Internal Nodes / Branches:** Intermediate nodes that have arrows pointing both toward and away from them [[02:50](http://www.youtube.com/watch?v=_L39rN6gz7Y&t=170)].
* **Leaf Nodes:** The final destination of a path where a classification is made [[03:00](http://www.youtube.com/watch?v=_L39rN6gz7Y&t=180)].

**Building a Tree:**
* The goal is to determine which feature (e.g., "Loves Popcorn," "Loves Soda," or "Age") should be the question at the top of the tree [[03:57](http://www.youtube.com/watch?v=_L39rN6gz7Y&t=237)].
* **Impurity:** A leaf is "impure" if it contains a mixture of different classifications [[05:43](http://www.youtube.com/watch?v=_L39rN6gz7Y&t=343)].
* **Gini Impurity:** A mathematical method used to quantify leaf impurity. Features with the **lowest weighted Gini Impurity** are chosen for splits [[06:28](http://www.youtube.com/watch?v=_L39rN6gz7Y&t=388)], [[12:15](http://www.youtube.com/watch?v=_L39rN6gz7Y&t=735)].

**Handling Different Data Types:**
* **Categorical Data:** Uses simple True/False splits [[04:15](http://www.youtube.com/watch?v=_L39rN6gz7Y&t=255)].
* **Numeric Data:** Involves **sorting** values and testing the **average** of adjacent values as potential thresholds to find the split with the lowest Gini Impurity [[10:00](http://www.youtube.com/watch?v=_L39rN6gz7Y&t=600)].

* **Preventing Overfitting:**
* To ensure the tree generalizes well to new data, techniques like **Pruning** or setting limits on the **minimum number of samples per leaf** are used [[16:10](http://www.youtube.com/watch?v=_L39rN6gz7Y&t=970)], [[16:36](http://www.youtube.com/watch?v=_L39rN6gz7Y&t=996)].

## **StatQuest: Decision Trees, Part 2 - Feature Selection and Missing Data**

https://www.youtube.com/watch?v=wpNl-JwwplA

This video covers how decision trees automatically handle extra information and what to do when your dataset has **holes** in it.

**Automatic Feature Selection:**
* Decision trees naturally decide which data is important. If a variable (like "chest pain") does not significantly reduce **impurity**, the algorithm will simply ignore it and not include it in the tree [[01:52](http://www.youtube.com/watch?v=wpNl-JwwplA&t=112)].
* By requiring a specific threshold of impurity reduction for each split, you can create simpler trees that avoid **overfitting** (performing well on training data but failing on new data) [[02:25](http://www.youtube.com/watch?v=wpNl-JwwplA&t=145)].

**Categorical Data:** If a value is missing, you can fill it using the **most common option** from that category or by using a **highly correlated column** to predict the likely answer [[03:27](http://www.youtube.com/watch?v=wpNl-JwwplA&t=207)].

**Numeric Data:** For missing numbers (like weight), you can substitute the **mean or median** [[04:19](http://www.youtube.com/watch?v=wpNl-JwwplA&t=259)]. Alternatively, you can identify a correlated variable (like height) and use **linear regression** to predict the missing value more accurately [[04:46](http://www.youtube.com/watch?v=wpNl-JwwplA&t=286)].

## Regression Trees, Clearly Explained!!!

http://www.youtube.com/watch?v=g9c66TUylZ4

This **StatQuest** video explains how **Regression Trees** work and how they are constructed to predict continuous numeric values.

* **Regression vs. Classification Trees:** While classification trees predict discrete categories, regression trees predict numeric values by using the average of the observations within a leaf [[02:17](http://www.youtube.com/watch?v=g9c66TUylZ4&t=137)].

* **Why Use Regression Trees?** They are useful when data doesn't follow a simple linear relationship (e.g., drug effectiveness that peaks at a moderate dosage but drops at high dosages) and can easily handle multiple predictors like age and sex alongside dosage [[01:28](http://www.youtube.com/watch?v=g9c66TUylZ4&t=88)], [[05:42](http://www.youtube.com/watch?v=g9c66TUylZ4&t=342)].

* **Threshold Selection:** The tree is built by testing different thresholds for each predictor. For numeric data, it calculates the **Sum of Squared Residuals (SSR)** for various potential split points [[10:50](http://www.youtube.com/watch?v=g9c66TUylZ4&t=650)].

* **The Root Node:** The predictor and threshold that result in the smallest SSR overall are chosen as the root of the tree [[12:49](http://www.youtube.com/watch?v=g9c66TUylZ4&t=769)].

* **Multiple Predictors:** If there are multiple predictors (e.g., dosage, age, sex), the model finds the best threshold for each and selects the one with the lowest SSR for the split [[19:04](http://www.youtube.com/watch?v=g9c66TUylZ4&t=1144)].

* **Preventing Overfitting:**
* To avoid "perfectly" fitting training data (which leads to high variance), a minimum number of observations required for a split is often set (e.g., 20 observations per node). If a node has fewer than this minimum, it becomes a leaf [[15:53](http://www.youtube.com/watch?v=g9c66TUylZ4&t=953)].
