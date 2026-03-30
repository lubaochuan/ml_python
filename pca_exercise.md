# Explore PCA

* Understand PCA as finding directions of **maximum variance**
* Interpret **principal components**
* Understand **dimensionality reduction**

Open: [https://numiqo.com/lab/pca](https://numiqo.com/lab/pca)

# Observing the Data

1. Describe the shape of the data

<details>
<summary>answer</summary>

Diagonal
</details>

2. If you had to summarize the data with a **single line**, what direction would you choose?

<details>
<summary>answer</summary>

Diagonal
</details>

3. Does PC1 align with the direction of greatest spread?
<details>
<summary>answer</summary>

Yes, because PCA finds the direction with **maximum variance**
</details>

4. What does PC1 represent in your own words?

<details>
<summary>answer</summary>

PC1 = direction capturing the most variation in data
</details>

5. Why might the direction of maximum variance be important?

<details>
<summary>answer</summary>

Because it preserves the most information
</details>

1. What is the angle between PC1 and PC2?

<details>
<summary>answer</summary>

90°
</details>

7. Why must PC2 be perpendicular to PC1?

<details>
<summary>answer</summary>

To ensure independence (no redundancy)
</details>

8. What kind of variation does PC2 capture?

<details>
<summary>answer</summary>

Remaining variation not captured by PC1
</details>

# Dimensionality Reduction

9. What happens to the data when projected onto PC1?

<details>
<summary>answer</summary>

Data is projected (flattened) onto a line
</details>

10.  What information is LOST?

<details>
<summary>answer</summary>

Variation perpendicular to PC1, fine-grained structure
</details>

11.  What information is PRESERVED?

<details>
<summary>answer</summary>

Main trend / largest variance
</details>

12. Why might we reduce dimensions? (check all that apply)

☐ Easier visualization

☐ Reduce noise

☐ Faster computation

☐ Increase complexity

<details>
<summary>answer</summary>

All but "increase complexity".
</details>

# Experimentation

### Experiment A: Circular Data

13. What happens to PC1 direction?

<details>
<summary>answer</summary>

Circular data → PC1 direction is unstable / arbitrary
</details>

### Experiment B: Noisy Data

14.  Does PCA still find the main direction?

<details>
<summary>answer</summary>

Yes, but may be slightly affected by noise
</details>

# Concept Check

15. PCA mainly:

    a) Minimizes error

    b) Maximizes variance

    c) Clusters data

    d) Sorts values

<details>
<summary>answer</summary>

b
</details>

1.  PC1:

    a) Captures least variance

    b) Captures most variance

    c) Is random

    d) Is always vertical

<details>
<summary>answer</summary>

b
</details>

1.  Principal components are perpendicular because:

    a) Easier to draw

    b) Avoid redundancy

    c) Required by software

    d) No reason

<details>
<summary>answer</summary>

b
</details>

# Final Reflection

18. In your own words, what problem does PCA solve?

<details>
<summary>answer</summary>

PCA finds the best directions (axes) to represent data by:

* Capturing maximum variance
* Reducing dimensions
* Preserving important structure
</details>