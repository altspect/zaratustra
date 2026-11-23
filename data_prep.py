import os
import torch
import torch.nn as nn
from torch.nn import functional as F

# Set manual seed for reproducibility
# This sets our "random" generated number algo to 1337, which means that every next number is going to be calculated from this algo
# based on number 1337
torch.manual_seed(1337)

# ==========================================
# PART 1: DATA PREPARATION & TOKENIZER
# ==========================================

# We create a blank variable, to which we are going to be inputting whole content of the books
text = ""
# We define a relative path to the folder
folder_path = "txts"

# Read all .txt files from the folder and combine them

# We iterate on files in folder, os.listdir tends to spit the files in random order, thats why we use sorted that lists them A to Z
for filename in sorted(os.listdir(folder_path)):
    # Only iterate on files that are of txt extnesion
    if filename.endswith(".txt"):
        # with open is a safe way to manage files in python, it means that the file is going to be deleted from the local memory
        # after we run the block of code. We use os.path.join to correctly fix the path, regardless of our os selection (windows/mac etc)
        # Parameter encoding='utf-8' is critical, there are many different letters and python needs to know how to enccde tem
        with open(os.path.join(folder_path, filename), 'r', encoding='utf-8') as f:
            # We read the whole content of the file and we merge it to our text var and add a new line character. It tends that the last
            # word of first file isn't merged with the first word of the last file
            text += f.read() + "\n"

print(f"Length of the combined text: {len(text):,} characters")

# Find all unique characters that occur in this text
# set is going to create a list of all unique chars
# list is going to change set which doesn't have order to list that has order (from {a, b, c, d} to [a, b, c, d])
# sorted is going to sort it alphabeticaly
chars = sorted(list(set(text)))
vocab_size = len(chars)

print(f"Vocabulary ready. Unique characters (vocab_size): {vocab_size}")
print(f"Vocabulary: {''.join(chars)}\n")

# Create a mapping from characters to integers, and vice versa
# enumerate is going to create an index for every element of the input [(0, ' '), (1, 'a'), (2, 'b'), (3, 'c'), ...]
# This is pretty much straightforward - first you create an object with character as a key and iterator as a value, and then other way around
# This will produce something like {'a': 1, 'b', 2, 'c': 3}
stoi = {ch: i for i, ch in enumerate(chars)}
# This will produce something like {1: 'a', 2: 'b', 3: 'c'}
itos = {i: ch for i, ch in enumerate(chars)}

# Encoder: takes a string, outputs a list of integers
# Will produce something like ala -> [1, 8, 1]
def encode(s):
    return [stoi[c] for c in s]

# Decoder: takes a list of integers, outputs a string
# Will produce something like [1, 8, 1] -> ala
def decode(l):
    return ''.join([itos[i] for i in l])


# ==========================================
# PART 2: DATASET & BATCHING
# ==========================================

# Encode the entire text dataset and store it in a PyTorch Tensor
# Will produce something like tensor([58, 1, 50, 1, 56, 59, ...])
data = torch.tensor(encode(text), dtype=torch.long)

# Split up the data into train and validation sets
n = int(0.9 * len(data)) # 90% for training, 10% for validation
train_data = data[:n]
val_data = data[n:]

## Hyperparameters for data loading
# Model is going to receive a table of numbers, and based on 8 previous is going to make another prediction
block_size = 8   # Maximum context length for predictions
batch_size = 4   # Number of independent sequences processed in parallel

def get_batch(split):
    # Select the correct dataset based on split argument - either train or val
    data_source = train_data if split == 'train' else val_data
    
    # Generate 4 random starting indices for our batch, will produce something like [1045, 8321, 203, 5521]
    ix = torch.randint(len(data_source) - block_size, (batch_size,))
    
    # x is the input context, y is the target (shifted by 1 character)
    x = torch.stack([data_source[i : i + block_size] for i in ix])
    y = torch.stack([data_source[i + 1 : i + block_size + 1] for i in ix])
    return x, y


# ==========================================
# PART 3: SELF ATTENTION LANGUAGE MODEL
# ==========================================

n_embd = 32

class Head(nn.Module):
    """Single head of self attention"""

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        # Register lower-triangular mask as a buffer (not a trainable model parameter)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)   # (B, T, head_size) - what information this token contains
        q = self.query(x) # (B, T, head_size) - what this token is looking for

        # Compute attention scores ("affinities") and scale by 1 / sqrt(head_size)
        wei = q @ k.transpose(-2, -1) * (k.shape[-1] ** -0.5) # (B, T, T)
        # Mask out future tokens so tokens only attend to the past (-inf becomes 0 after softmax)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1) # (B, T, T)

        # Perform the weighted aggregation of the values
        v = self.value(x) # (B, T, head_size)
        out = wei @ v     # (B, T, head_size)
        return out
    
class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        # Each token directly reads off the embedding vector from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        # Positional embeddings encode where each token sits in the block_size window
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.sa_head = Head(n_embd)
        # Linear layer projecting embeddings back to vocabulary logits
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # idx and targets are both (B, T) tensors of integers
        tok_emb = self.token_embedding_table(idx) # (B, T, n_embd)
        pos_emb = self.position_embedding_table(torch.arange(T)) # (T, n_embd)
        x = tok_emb + pos_emb # (B, T, n_embd) - combine token identity with position
        x = self.sa_head(x)   # (B, T, n_embd) - apply one head of self-attention
        logits = self.lm_head(x) # (B, T, vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # Crop idx to the last block_size tokens so position_embedding_table doesn't go out of bounds
            idx_cond = idx[:, -block_size:]
            # Get the predictions
            logits, loss = self(idx_cond)
            # Focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # Apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # Sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # Append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx


# # ==========================================
# # PART 4: TESTING THE MODEL
# # ==========================================

# 1. Initialize the model
m = BigramLanguageModel(vocab_size)

# 2. Get a batch of data
xb, yb = get_batch('train')

# 3. Forward pass
logits, loss = m(xb, yb)

print("--- MODEL INITIALIZATION ---")
print(f"Logits shape: {logits.shape}")
print(f"Initial loss (before training): {loss.item():.4f}\n")

# 4. Generate untrained text
print("--- UNTRAINED GENERATION (Should be random noise) ---")
# Start with a 1x1 tensor holding a zero (often maps to a newline or space)
context = torch.zeros((1, 1), dtype=torch.long)
generated_tokens = m.generate(context, max_new_tokens=100)[0].tolist()

print(decode(generated_tokens))

# ==========================================
# PART 5: TRAINING (Filling in our lookup table!)
# ==========================================

# Optimizer that will adjust the weights in the lookup table
optimizer = torch.optim.AdamW(m.parameters(), lr=1e-2)

batch_size = 32
steps = 10000

print(f"--- STARTING TRAINING ({steps} steps) ---")
for step in range(steps):
    # 1. Sample a batch of data
    xb, yb = get_batch('train')

    # 2. Evaluate the loss
    logits, loss = m(xb, yb)

    # 3. Update the weights in the matrix
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    # Print progress every 1000 steps
    if step % 1000 == 0 or step == steps - 1:
        print(f"Step {step:5d} | Loss: {loss.item():.4f}")

# ==========================================
# PART 6: POST-TRAINING GENERATION
# ==========================================
print("\n--- GENERATION AFTER TRAINING ---")
context = torch.zeros((1, 1), dtype=torch.long)
print(decode(m.generate(context, max_new_tokens=400)[0].tolist()))