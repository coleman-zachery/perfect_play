There are several scenarios where one-hot encoding everything becomes inefficient or counterproductive, especially in deep learning or large combinatorial problems. Let’s go through them carefully.

1. High-dimensional categorical variables

Problem: One-hot encoding creates a new dimension for every possible category.

Example: Encoding a deck of 52 cards as one-hot for each card in a hand → 52×13 = 676 dimensions if you use one-hot per card.

Neural networks now have to learn in a very sparse, high-dimensional space, which is inefficient.

Why it’s bad:

Large input vectors slow training and increase memory usage.

Sparse representations make learning relationships between categories harder.

2. Variables with natural relationships

If a variable has intrinsic structure, one-hot encoding destroys it:

Ordinal data: e.g., ranks, age groups, levels.

One-hot ignores order, so the network won’t automatically know that “3 < 4 < 5.”

Continuous-like variables: e.g., temperature ranges or numeric scores.

One-hot splits them into unrelated categories, losing numeric continuity.

Better approach:

Use integer + embeddings for categorical variables.

Use raw numeric values for continuous or ordinal variables.

3. When you have combinatorial input

Example: Encoding a chess board or Go board by one-hoting all piece types and positions.

The vector explodes in size (e.g., 64×12 = 768 dimensions for chess), making the network unnecessarily large.

Embeddings or structured representations (like 8×8×12 tensor) are much more efficient.

4. When relationships between categories matter

One-hot treats categories as independent, so the network has to learn correlations from scratch.

Example: in bridge, one-hot encoding each card gives no indication that all kings are similar or that suits are related.

Using embeddings lets the network learn these relationships efficiently.

5. Sparse inputs with deep networks

If most of your one-hot inputs are 0 (common with large categorical variables), a dense fully-connected network will waste computation processing zeros.

Better alternatives:

Embeddings

Sparse matrix operations

Convolutional or attention-based architectures that exploit structure

✅ Summary / Rules of Thumb

Use one-hot encoding when:

Categories are small (e.g., 4 suits, 13 ranks individually if needed).

There is no ordinal or structural relationship you want the model to exploit.

Avoid one-hot when:

Categories are high-dimensional or combinatorial (e.g., full deck of cards).

There are meaningful relationships or order between categories.

Sparsity would make training inefficient.