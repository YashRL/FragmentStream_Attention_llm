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
