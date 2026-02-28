# Chapter 3 – Real-world AI

* Why do real-world AI systems need to deal with uncertainty?

<details>
<summary>answer</summary>

Because real-world information is often incomplete, noisy, or ambiguous, AI systems cannot rely on perfect information and must reason under uncertainty.
</details>

* Why did early rule-based AI systems struggle in real-world environments?

<details>
<summary>answer</summary>

Early AI systems assumed precise rules and exact information, which does not match the uncertainty and variability found in real-world situations.
</details>

* What role does probability play in real-world AI?

<details>
<summary>answer</summary>

Probability allows AI systems to quantify uncertainty and make informed decisions even when outcomes are not guaranteed.
</details>

* Why can’t we evaluate probabilistic predictions based on a single outcome?

<details>
<summary>answer</summary>

Because probabilities describe long-term behavior, a single event may occur even if it is unlikely, so accuracy must be evaluated over many cases.
</details>

* What is the difference between probability and odds?

<details>
<summary>answer</summary>

Probability describes how often an event occurs out of all possibilities, while odds compare how often the event occurs versus how often it does not.
</details>

* What is Bayes’ rule used for in this chapter?

<details>
<summary>answer</summary>

Bayes’ rule is used to update existing beliefs when new evidence becomes available.
</details>

* What are prior odds and posterior odds?

<details>
<summary>answer</summary>

Prior odds represent beliefs before seeing new evidence, while posterior odds represent updated beliefs after considering the evidence.
</details>

* Why is Bayes’ rule powerful despite being mathematically simple?

<details>
<summary>answer</summary>

Because it provides a systematic way to combine multiple pieces of evidence and avoid intuitive reasoning errors.
</details>

* What is the Naive Bayes classifier used for?

<details>
<summary>answer</summary>

Naive Bayes is used to classify items, such as emails, by estimating probabilities based on observed features.
</details>

* Why is the Naive Bayes classifier described as “naive”?

<details>
<summary>answer</summary>

Because it assumes that features are independent given the class, which is a simplification of reality.
</details>

* How does Naive Bayes use multiple pieces of evidence?

<details>
<summary>answer</summary>

It combines evidence by multiplying likelihood ratios for each observed feature.
</details>

* Why does Naive Bayes often work well in practice despite its simplifying assumptions?

<details>
<summary>answer</summary>

Because capturing the most important statistical patterns is often sufficient for good performance.
</details>

* Why is probabilistic reasoning better than intuition in uncertain situations?

<details>
<summary>answer</summary>

Probabilistic reasoning accounts for all relevant information systematically, while intuition often ignores base rates and uncertainty.
</details>
