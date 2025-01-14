# **FragmentStream Language Model (Custom Attention Mechanism)**

This repository provides an implementation of a transformer-based language model designed to efficiently handle long sequences with a custom attention mechanism called **FragmentStream_Attention**. The model leverages **Fragmented Attention** to reduce memory usage, inspired by the concept of flash attention but optimized for GPUs with lower resources.

## **Overview**

The model is based on the transformer architecture and uses a custom attention mechanism, designed to process attention in smaller fragments, which allows it to handle larger sequences without running into memory limitations. Additionally, the model includes a multi-head attention mechanism, feed-forward layers, and uses various activation functions for experimentation. It also includes support for generating text with temperature-based sampling.

---

## **Features**

- **Fragmented Attention**: Memory-efficient attention mechanism that processes queries and keys in fragments instead of all at once, optimizing GPU memory usage.
- **Dynamic Multi-Head Attention**: The number of attention heads is adjusted based on the sequence length.
- **Feed-Forward Layers**: Fully connected layers with dynamic activation functions.
- **Model Generation**: A text generation method with temperature-based sampling and stopping conditions based on a special stop token.
- **Tokenizer**: A Byte-Pair Encoding (BPE)-based tokenizer that is trained on the provided dataset.
- **Training**: A comprehensive training loop with checkpoint saving and model evaluation.

---

**FragmentStream_Attention** is an innovative attempt at replicating **flash attention** from scratch while optimizing memory usage, especially for those without access to cutting-edge GPUs like Aphere or Hopper architectures. By processing attention in small, manageable fragments instead of computing the full attention matrix at once, **FragmentStream_Attention** enables training large transformer models on consumer-grade GPUs without overwhelming memory resources. 

This repository presents an efficient and scalable approach to implementing transformers using the **FragmentStream Attention** mechanism, ideal for use cases that require memory optimization.

---

## Key Features

### 1. **FragmentStream Attention (Key Innovation)**
- **Memory Efficient:** FragmentStream Attention processes attention in small chunks (128 tokens at a time), significantly reducing memory consumption compared to the traditional O(n²) memory complexity of full attention.
- **Optimized for Consumer GPUs:** With this mechanism, transformer models can be trained on GPUs with less memory, making it more accessible for smaller-scale setups.
  
### 2. **Adaptive Architecture**
- **Dynamic Head Selection:** The architecture dynamically adjusts the number of attention heads based on the sequence length, optimizing memory usage and computation.
- **Pre-LayerNorm:** Pre-LayerNorm architecture is used to improve training stability and enhance model performance.

### 3. **Efficient Training on Consumer Hardware**
- This model has been successfully trained using Kaggle’s P100 GPU and achieved strong results with significantly reduced computational requirements.
  
---

## Implementation Overview

Certainly! Here’s how you can structure this section in your README, emphasizing the comparison between traditional attention and the FragmentStream_Attention:

---

### **Traditional Attention vs FragmentStream_Attention**

In traditional transformer models, attention is computed in a way that requires storing the entire attention matrix in memory, which becomes impractical when dealing with long sequences. Here's a simplified breakdown:

#### **Traditional Attention (Simplified)**

In the traditional approach, we compute attention over the entire sequence at once. For each input tensor, we calculate queries (`q`), keys (`k`), and values (`v`) from the input sequence:

```python
B, T, C = x.shape  # B = batch size, T = sequence length, C = number of features
q = self.query(x)  # (B, T, C)
k = self.key(x)    # (B, T, C)
```

Now, we calculate the attention scores between all queries and keys using a matrix multiplication:

```python
attention_scores = q @ k.transpose(-2, -1)  # (B, T, T) - This is huge!
attention = softmax(attention_scores) @ v    # More memory usage
```

In this process, we calculate the full attention matrix, which has a shape of `(B, T, T)`. As the sequence length (`T`) grows, the memory requirements increase quadratically (`O(T²)`), making it difficult to scale this approach for long sequences.

#### **FragmentStream_Attention (Optimized Approach)**

In contrast, **FragmentStream_Attention** optimizes this by breaking the attention computation into smaller fragments, greatly reducing memory usage. Instead of processing the entire attention matrix, we divide the sequence into smaller chunks and process them independently.

Here’s how the **FragmentStream_Attention** approach works:

```python
fragment_size = 128  # Process 128 tokens at a time
for i in range(0, T, fragment_size):  # Process queries in fragments
    q_fragment = q[:, i:i+fragment_size]  # Take a small group of queries
    for j in range(0, T, fragment_size):  # Process keys/values in fragments
        k_fragment = k[:, j:j+fragment_size]  # Take a small group of keys
        v_fragment = v[:, j:j+fragment_size]  # And the corresponding values        
        # Compute attention only on these small fragments
        scores = q_fragment @ k_fragment.transpose(-2, -1)
        # Process and accumulate results
```

This method reduces memory consumption by only keeping a small portion of the attention matrix in memory at any given time.

#### **Visualizing the Process**

Here’s an example to visualize the difference:

1. **Traditional Attention (Full Matrix in Memory)**

    The full attention matrix is computed and stored in memory. This consumes a large amount of memory, especially as `T` increases.

    ```
    [Full Matrix in Memory]
    X X X X X X X X X X
    X X X X X X X X X X
    X X X X X X X X X X
    X X X X X X X X X X
    ```

2. **FragmentStream_Attention (Fragments Processed Sequentially)**

    In **FragmentStream_Attention**, we process small fragments of the sequence, cleaning up each fragment after it’s processed before moving on to the next one:

    ```
    [fragment 1]   [Clean Up]   [fragment 2]   [Clean Up]
    X X X ➜ X X X ➜ X X X ➜ X X X
    X X X ➜ X X X ➜ X X X ➜ X X X
    ```

    This reduces memory requirements significantly because we only keep a small part of the attention matrix in memory at any given moment.

---

By breaking down the attention computation into smaller pieces, **FragmentStream_Attention** achieves memory efficiency, allowing the model to process long sequences that would otherwise exceed the memory capacity of most GPUs.

---

Let me know if you'd like to adjust or expand any part of this explanation further!

# Yes It may sound funny but it make signifact changes


**Code Snippet Example (FragmentStream_Attention):**
```python
class FragmentStream_Attention(nn.Module):
    def __init__(self, head_size, block_size, dropout):
        super().__init__()
        self.head_size = head_size
        self.fragment_size = 128  # Adjust based on your GPU memory
        self.dropout = nn.Dropout(dropout)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        
    def forward(self, q, k, v):
        B, T, C = q.shape
        out = torch.zeros_like(v)

        for i in range(0, T, self.fragment_size):
            j_start = i
            j_end = min(T, i + self.fragment_size)
            q_fragment = q[:, i:j_end]

            attn_weights = torch.zeros(B, j_end-i, T, device=q.device)
            
            for j in range(0, T, self.fragment_size):
                k_fragment = k[:, j:min(T, j + self.fragment_size)]
                scores = (q_fragment @ k_fragment.transpose(-2, -1)) * (C ** -0.5)
                
                scores = scores.masked_fill(self.tril[i:j_end, j:min(T, j + self.fragment_size)] == 0, float('-inf'))
                attn_weights[:, :, j:min(T, j + self.fragment_size)] = scores

            attn_weights = F.softmax(attn_weights, dim=-1)
            attn_weights = self.dropout(attn_weights)

            for j in range(0, T, self.fragment_size):
                v_fragment = v[:, j:min(T, j + self.fragment_size)]
                out[:, i:j_end] += attn_weights[:, :, j:min(T, j + self.fragment_size)] @ v_fragment

        return out
```

---

## Practical Results

**Dataset:**
- The model was successfully trained on a healthcare dataset using Kaggle’s P100 GPU.

**Results:**
- The model demonstrated:
  - **Coherent medical responses** when queried.
  - A strong **understanding of medical contexts**.
  - **Lower computational overhead** compared to traditional transformer architectures, making it suitable for environments with limited resources.

---

## Open Source & Community

This project is open source and aims to contribute to the community by providing a scalable, memory-efficient transformer model that can be easily adapted for various applications.

- **Use Cases:**
  - Memory-efficient transformers for large sequence data.
  - Experimentation with novel attention mechanisms in NLP, vision, and other fields.
  - Developers working with constrained hardware.

- **Key Contributions:**
  - FragmentStream Attention for improved memory usage.
  - Adaptive head selection for better efficiency based on input length.
  - Full implementation of transformer architecture optimized for GPUs with limited VRAM.

---

## Conclusion

**FragmentStream_Attention** introduces a novel method for efficiently processing large sequences with transformers. It reduces memory requirements by processing attention in small fragments, making it a practical solution for developers working with limited GPU memory. If you have any questions or wish to contribute to the project, feel free to open an issue or submit a pull request!

Happy experimenting!
---

## **Installation**

To run this model, ensure you have the following dependencies installed:

```bash
pip install torch tokenizers
```

---

## **Usage**

### 1. **Preparing Data**

This model works with JSON files that contain Question-Answer pairs. Ensure your training and test data are in the following format:

```json
[
    {
        "Question": "What is AI?",
        "Answer": "Artificial Intelligence (AI) is intelligence demonstrated by machines."
    },
    {
        "Question": "What is NLP?",
        "Answer": "Natural Language Processing (NLP) is a subfield of linguistics, computer science, and artificial intelligence concerned with the interactions between computers and human (natural) languages."
    }
]
```

### 2. **Training the Model**

To train the model, you need to provide paths to the training and testing JSON files. Optionally, you can also resume from a checkpoint.

```python
train_path = "/path/to/your/training_data.json"
test_path = "/path/to/your/test_data.json"
resume_checkpoint = "/path/to/your/checkpoint_file.yash"  # optional

model, tokenizer = train_model(train_path, test_path, resume_checkpoint)
```

The model will train and periodically save checkpoints. It also evaluates the model at each `eval_interval` and saves the best model based on the test loss.

### 3. **Text Generation**

Once the model is trained, you can use it to generate text. Here is an example of how to generate text from the model:

```python
# Use the tokenizer and model for generation
idx = tokenizer.encode("What is the capital of France?").ids  # Example prompt
idx = torch.tensor(idx).unsqueeze(0).to(device)

generated_tokens = model.generate(idx, max_new_tokens=50, tokenizer=tokenizer)
generated_text = tokenizer.decode(generated_tokens[0].tolist())
print(generated_text)
```

This will generate a response to the question "What is the capital of France?" based on the training data.

---

## **Model Architecture**

The core architecture is based on the Transformer with some customizations:

### 1. **FragmentStream_Attention**:
   - A custom attention mechanism that splits the attention computation into smaller chunks (fragments) to save memory.
   
### 2. **MultiHeadAttention**:
   - The attention heads are dynamically selected based on the sequence length. The model can use different numbers of attention heads for shorter and longer sequences.

### 3. **FeedForward Layer**:
   - A fully connected layer with dynamic activation functions (ReLU, GELU, Leaky ReLU, or SiLU).

### 4. **Block**:
   - Each block consists of a multi-head attention layer, followed by a feed-forward layer with layer normalization applied both before and after the attention and feed-forward layers.

### 5. **Language Model**:
   - The final language model consists of an embedding layer, several transformer blocks, layer normalization, and an output linear layer for token predictions.

---

## **Training Details**

- **Hyperparameters**:
    - `batch_size`: 32
    - `dropout`: 0.125
    - `learning_rate`: 3e-4
    - `max_iters`: 500,000
    - `block_size`: 512
    - `n_embd`: 128 (embedding size)
    - `n_head`: 8 (number of attention heads)
    - `n_layer`: 8 (number of transformer blocks)

- **Optimizer**: AdamW
- **Learning Rate Scheduler**: Cosine Annealing
- **Evaluation**: Every 50 iterations, the model is evaluated on the training and test datasets.

---

## **Custom Attention Mechanism: FragmentStream_Attention**

The **FragmentStream_Attention** mechanism divides the attention computation into smaller fragments to reduce the memory footprint. This is particularly useful when working with long sequences, as traditional attention mechanisms scale quadratically with sequence length. By using fragmented attention, this model can process long sequences efficiently.

### **How It Works**:

- The attention is computed in smaller fragments, each of size `fragment_size` (e.g., 128 tokens).
- For each fragment of queries, we compute attention with each fragment of keys and values.
- We apply a causal mask to prevent attention to future tokens.
- The results from each fragment are accumulated to form the final attention output.

---

## **Performance Optimization**

This model is designed to run efficiently on GPUs with limited VRAM by using fragmented attention. The memory usage of the model is optimized by splitting the attention computation into smaller fragments rather than processing the entire sequence at once. Additionally, the number of attention heads is adaptive, so the model can dynamically adjust based on the input sequence length.

---

## **Important Notes**

- **GPU Usage**: Ensure you are running the model on a machine with a compatible GPU for training. If you're using a CPU, the training time will be significantly longer.
- **Flash Attention**: The code also includes a reference to "Flash Attention," which could be used if the GPU and PyTorch version support it. However, the custom **FragmentStream_Attention** is provided as a fallback for older hardware.
- **Checkpointing**: The model periodically saves checkpoints. You can resume training from the latest checkpoint if needed.

---

## **License**

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## **Acknowledgments**

- Inspired by Andrej Karpathy's ideas on beautiful numbers and transformer models.
- Flash attention concept was referenced as inspiration for this model's custom attention mechanism.

--- 

This README provides a complete overview of your model, instructions for setup, and guidance on using the code. Let me know if you need any further clarifications!
