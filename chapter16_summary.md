# Chapter 16 Summary: Natural Language Processing with RNNs and Attention

Natural Language Processing (NLP) focuses on enabling machines to understand and generate human language.

This chapter covers:

* Text preprocessing
* Word representations (embeddings)
* Recurrent Neural Networks (RNNs)
* Sequence-to-sequence models
* Attention mechanisms

Language is sequential → models must capture **order and context**

## 1. Text Representation

### Tokenization

* Splitting text into words or subwords

Example:

```id="s0hcdl"
"I love AI" → ["I", "love", "AI"]
```

### Encoding Methods

#### One-Hot Encoding

* Each word → sparse vector
* Large vocabulary → inefficient

#### Word Embeddings

* Dense vectors capturing meaning
* Similar words → similar vectors

Example:

* “king” − “man” + “woman” ≈ “queen”

## 2. Word Embeddings

* Learned during training or pre-trained
* Capture semantic relationships

Popular methods:

* Word2Vec
* GloVe

Embeddings reduce dimensionality and improve generalization

## 3. Recurrent Neural Networks (RNNs)

Designed for sequential data:

* Maintain **hidden state**
* Process inputs step-by-step

$$
h_t = f(x_t, h_{t-1})
$$

### RNN Challenges

* Vanishing gradients
* Difficulty learning long-term dependencies

## 4. LSTM and GRU (Improved RNNs)

### LSTM (Long Short-Term Memory)

* Uses gates:

  * Forget gate
  * Input gate
  * Output gate

### GRU (Gated Recurrent Unit)

* Simpler than LSTM
* Similar performance

Both help capture long-range dependencies

## 5. Sequence-to-Sequence Models (Seq2Seq)

Used for tasks like:

* Machine translation
* Text summarization

### Architecture:

* Encoder → processes input sequence
* Decoder → generates output sequence

## 6. Limitation of Basic Seq2Seq

* Encoder compresses entire input into a single vector
* Information bottleneck for long sequences

## 7. Attention Mechanism

### Key Idea:

Allow the model to **focus on relevant parts of the input**

Instead of: One fixed vector

Use: Weighted combination of all encoder states

### Attention Process

1. Compute alignment scores
2. Convert to weights (softmax)
3. Compute context vector
4. Use context for prediction

## 8. Benefits of Attention

* Handles long sequences better
* Improves performance
* Provides interpretability (which words matter)

## 9. Implementing NLP Models in Keras

### Example: RNN Layer

```python
keras.layers.SimpleRNN(50)
```

### LSTM

```python
keras.layers.LSTM(50)
```

### GRU

```python
keras.layers.GRU(50)
```

## 10. Text Classification

Pipeline:

1. Tokenize text
2. Convert to sequences
3. Pad sequences
4. Feed into RNN

## 11. Preprocessing Tools

* `Tokenizer`
* `pad_sequences()`

Ensures uniform input length

## 12. Beyond RNNs

Attention leads to:

* Transformers (next chapter)
* More scalable NLP models

# Glossary

| Term               | Definition                                 |
| ------------------ | ------------------------------------------ |
| NLP                | Field focused on processing human language |
| Tokenization       | Splitting text into units                  |
| One-Hot Encoding   | Sparse representation of words             |
| Word Embedding     | Dense vector representation of words       |
| RNN                | Neural network for sequential data         |
| Hidden State       | Memory of previous inputs                  |
| Vanishing Gradient | Gradients shrink during training           |
| LSTM               | RNN variant with memory gates              |
| GRU                | Simpler gated RNN                          |
| Seq2Seq            | Encoder-decoder architecture               |
| Attention          | Mechanism to focus on relevant input       |
| Context Vector     | Weighted representation of input           |
| Padding            | Making sequences equal length              |
| Decoder            | Generates output sequence                  |
| Encoder            | Processes input sequence                   |

# Review Questions

### 1. Why is sequential modeling important in NLP?

<details>
<summary>Answer</summary>

Because word order affects meaning, and models must capture context across time.
</details>

### 2. What is the limitation of one-hot encoding?

<details>
<summary>Answer</summary>

It creates high-dimensional sparse vectors and does not capture relationships between words.
</details>

### 3. What advantage do word embeddings provide?

<details>
<summary>Answer</summary>

They capture semantic similarity and reduce dimensionality.
</details>

### 4. Why do standard RNNs struggle with long sequences?

<details>
<summary>Answer</summary>

Due to vanishing gradients, they forget earlier information.
</details>

### 5. How do LSTMs solve this problem?

<details>
<summary>Answer</summary>

They use gates to control information flow and preserve long-term dependencies.
</details>

### 6. What is the role of the hidden state in an RNN?

<details>
<summary>Answer</summary>

It stores information from previous time steps and passes it forward.
</details>

### 7. What is the purpose of padding sequences?

<details>
<summary>Answer</summary>

To ensure all inputs have the same length for batch processing.
</details>

### 8. Why is attention better than a single context vector?

<details>
<summary>Answer</summary>

It allows the model to dynamically focus on different parts of the input.
</details>

### 9. What is the encoder-decoder structure used for?

<details>
<summary>Answer</summary>

Transforming one sequence into another (e.g., translation).
</details>

### 10. How does attention improve interpretability?

<details>
<summary>Answer</summary>

It shows which input tokens the model focuses on during prediction.
</details>

### 11. What does this layer do?

```python
keras.layers.LSTM(50)
```

<details>
<summary>Answer</summary>

Creates an LSTM layer with 50 units to process sequences.
</details>

### 12. What happens during tokenization?

<details>
<summary>Answer</summary>

Text is converted into integer sequences representing words.
</details>

### 13. What does `pad_sequences()` do?

<details>
<summary>Answer</summary>

It ensures all sequences have equal length by adding padding tokens.
</details>

### 14. Why does attention eliminate the bottleneck in Seq2Seq models?

<details>
<summary>Answer</summary>

Because instead of compressing all information into a single vector, attention allows the decoder to access **all encoder states**, dynamically selecting relevant information.
</details>
