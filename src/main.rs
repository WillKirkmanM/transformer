use ndarray::{arr2, Array, Array2, Axis};

// A softmax function for normalisation
fn softmax(matrix: &Array2<f32>) -> Array2<f32> {
    let max = matrix.fold(f32::NEG_INFINITY, |acc, &x| acc.max(x));
    let exp_matrix = matrix.mapv(|x| (x - max).exp());
    let sum_exp = exp_matrix.sum_axis(Axis(1)).insert_axis(Axis(1));
    exp_matrix / sum_exp
}

/// Represents the Multi-Head Attention mechanism.
/// In a real model, Wq, Wk, and Wv would be learned weights.
/// Here, we use identity matrices.
struct MultiHeadAttention {
    w_q: Array2<f32>, // Query weight matrix
    w_k: Array2<f32>, // Key weight matrix
    w_v: Array2<f32>, // Value weight matrix
    d_k: f32,       // Dimension of the key vectors, for scaling
}

impl MultiHeadAttention {
    fn new(embedding_dim: usize) -> Self {
        Self {
            // In a real scenario, these would be initialised with random weights.
            w_q: Array::eye(embedding_dim),
            w_k: Array::eye(embedding_dim),
            w_v: Array::eye(embedding_dim),
            d_k: (embedding_dim as f32).sqrt(),
        }
    }

    // The forward pass of the attention mechanism
    fn forward(&self, input: &Array2<f32>) -> Array2<f32> {
        // 1. Create Q, K, V matrices from the input
        let q = input.dot(&self.w_q);
        let k = input.dot(&self.w_k);
        let v = input.dot(&self.w_v);

        // 2. Calculate attention scores: Q * K^T / sqrt(d_k)
        let scores = q.dot(&k.t()) / self.d_k;
        println!("Attention Scores (before softmax):\n{:.2}\n", scores);

        // 3. Apply softmax to get attention weights
        let weights = softmax(&scores);
        println!("Attention Weights (after softmax):\n{:.2}\n", weights);

        // 4. Multiply weights by V to get the output
        weights.dot(&v)
    }
}

/// Represents the simple Feed-Forward Network.
struct FeedForward {
    w1: Array2<f32>,
    w2: Array2<f32>,
}

impl FeedForward {
    fn new(embedding_dim: usize, ff_hidden_dim: usize) -> Self {
        // Using identity-like matrices for a direct passthrough demonstration
        let mut w1 = Array::zeros((embedding_dim, ff_hidden_dim));
        for i in 0..embedding_dim {
            if i < ff_hidden_dim {
                w1[[i, i]] = 1.0;
            }
        }

        let mut w2 = Array::zeros((ff_hidden_dim, embedding_dim));
        for i in 0..embedding_dim {
            if i < ff_hidden_dim {
                w2[[i, i]] = 1.0;
            }
        }
        
        Self { w1, w2 }
    }

    // A simple forward pass with a ReLU-like activation (max(0, x))
    fn forward(&self, input: &Array2<f32>) -> Array2<f32> {
        let hidden = input.dot(&self.w1).mapv(|x| x.max(0.0)); // ReLU
        hidden.dot(&self.w2)
    }
}

/// Represents a single Encoder Layer of a Transformer.
struct EncoderLayer {
    attention: MultiHeadAttention,
    feed_forward: FeedForward,
}

impl EncoderLayer {
    fn new(embedding_dim: usize, ff_hidden_dim: usize) -> Self {
        Self {
            attention: MultiHeadAttention::new(embedding_dim),
            feed_forward: FeedForward::new(embedding_dim, ff_hidden_dim),
        }
    }

    // The forward pass through the entire layer
    fn forward(&self, input: &Array2<f32>) -> Array2<f32> {
        // 1. Pass through multi-head attention
        // In a real implementation, includes residual connection & layer norm
        let attention_output = self.attention.forward(input);

        // 2. Pass through feed-forward network
        // Also includes residual connection & layer norm
        self.feed_forward.forward(&attention_output)
    }
}

fn main() {
    let embedding_dim = 4; // Dimension of the word embeddings
    let sequence_length = 3; // Number of "words" in the sentence

    // Let's create a sample input sequence (3 words, 4 dimensions each)
    // Think of this as: ["hello", "world", "!"] after embedding.
    let input = arr2(&[
        [1.0, 0.5, 0.2, 0.1], // Embedding for "hello"
        [0.6, 1.2, 0.8, 0.3], // Embedding for "world"
        [0.1, 0.3, 0.7, 1.1], // Embedding for "!"
    ]);

    println!("Input Sequence ({} words, {} dims):\n{:.2}\n", sequence_length, embedding_dim, input);

    // Initialise the encoder layer
    let encoder = EncoderLayer::new(embedding_dim, 8); // Hidden dim for FF network is 8

    // Run the forward pass
    let output = encoder.forward(&input);

    println!("Final Output of Encoder Layer:\n{:.2}", output);
}