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
chars = sorted(list(set(text)))
vocab_size = len(chars)

print(f"Vocabulary ready. Unique characters (vocab_size): {vocab_size}")
print(f"Vocabulary: {''.join(chars)}\n")

# Create a mapping from characters to integers, and vice versa
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}

# Encoder: takes a string, outputs a list of integers
def encode(s):
    return [stoi[c] for c in s]

# Decoder: takes a list of integers, outputs a string
def decode(l):
    return ''.join([itos[i] for i in l])


# ==========================================
# PART 2: DATASET & BATCHING
# ==========================================

# Encode the entire text dataset and store it in a PyTorch Tensor
data = torch.tensor(encode(text), dtype=torch.long)

# Split up the data into train and validation sets
n = int(0.9 * len(data)) # 90% for training, 10% for validation
train_data = data[:n]
val_data = data[n:]

# Hyperparameters for data loading
block_size = 8   # Maximum context length for predictions
batch_size = 4   # Number of independent sequences processed in parallel

def get_batch(split):
    # Select the correct dataset
    data_source = train_data if split == 'train' else val_data
    
    # Generate random starting indices for our batch
    ix = torch.randint(len(data_source) - block_size, (batch_size,))
    
    # x is the input context, y is the target (shifted by 1 character)
    x = torch.stack([data_source[i : i + block_size] for i in ix])
    y = torch.stack([data_source[i + 1 : i + block_size + 1] for i in ix])
    return x, y


# ==========================================
# PART 3: BIGRAM LANGUAGE MODEL
# ==========================================

class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        # Lookup table reading logits for the next token directly
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        # idx and targets are both (B, T) tensor of integers
        # B = Batch Size (4), T = Time/Block Size (8), C = Vocab Size (e.g., 65)
        
        logits = self.token_embedding_table(idx) # Shape: (B, T, C)
        
        if targets is None:
            loss = None
        else:
            # PyTorch expects dimensions to be (Batch * Time, Channels) for cross_entropy
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            
            # Calculate cross-entropy loss (how wrong the predictions are)
            loss = F.cross_entropy(logits, targets)
            
        return logits, loss
        
    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # 1. Get the predictions
            logits, loss = self(idx)
            
            # 2. Focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            
            # 3. Apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            
            # 4. Sample from the distribution to get the next character
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            
            # 5. Append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
            
        return idx


# ==========================================
# PART 4: TESTING THE MODEL
# ==========================================

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