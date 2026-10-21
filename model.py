import torch
import torch.nn as nn
import torch.nn.functional as F

# Self attention head for predicting next character.
class Head(nn.Module):
    def __init__(self, vector_size, head_size, block_size, dropout):
        super().__init__()
        self.key = nn.Linear(vector_size, head_size, bias=False) # What each token uses to match with a query
        self.query = nn.Linear(vector_size, head_size, bias=False) # What each token is looking for
        self.value = nn.Linear(vector_size, head_size, bias=False) # What information each token shares

        # Mask to prevent model from 'looking ahead' to prevent it from just copying the next letter in x
        self.register_buffer("mask", torch.tril(torch.ones(block_size, block_size)))

        # Dropout layer to prevent overfitting
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        _, seq_len, _ = x.shape # seq_len is how many characters are in the input
        key, query, value = self.key(x), self.query(x), self.value(x)
        scores = query @ key.transpose(-2, -1) * key.shape[-1] ** -0.5 # scaled match score for every pair of positions
        scores = scores.masked_fill(self.mask[:seq_len, :seq_len] == 0, float("-inf")) # block the answer
        weights = self.dropout(F.softmax(scores, dim=-1)) # scores -> probabilities
        return weights @ value # weighted mix of values

# Runs several heads in parallel and combines their outputs
class MultiHeadAttention(nn.Module):
    def __init__(self, vector_size, num_heads, block_size, dropout):
        super().__init__()
        head_size = vector_size // num_heads
        self.heads = nn.ModuleList([Head(vector_size, head_size, block_size, dropout) for _ in range(num_heads)]) # num_heads heads for paying attention to different information
        self.projection = nn.Linear(vector_size, vector_size) # Projection for combining all heads results
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([head(x) for head in self.heads], dim=-1) # Concat each heads result for x side by side (batch size, sequence length, vector_size) 
        return self.dropout(self.projection(out))

# Feed forward for pattern detection in learned attention
class FeedForward(nn.Module):
    def __init__(self, vector_size, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(vector_size, 4 * vector_size), # expand for more room to detect patterns
            nn.ReLU(), # nonlinearity
            nn.Linear(4 * vector_size, vector_size), # shrink back to original
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

# Blocks learn relationships between tokens
class Block(nn.Module):
    def __init__(self, vector_size, num_heads, block_size, dropout):
        super().__init__()
        self.attention = MultiHeadAttention(vector_size, num_heads, block_size, dropout) # Multiple heads for keeping attention to various types of information
        self.feedforward = FeedForward(vector_size, dropout) # Extract patterns from learned attention
        self.ln1 = nn.LayerNorm(vector_size) # Scale so numbers stay reasonable
        self.ln2 = nn.LayerNorm(vector_size)

    def forward(self, x):
        x = x + self.attention(self.ln1(x)) # Add context from earlier tokens
        x = x + self.feedforward(self.ln2(x)) # Extract patterns from gathered context
        return x


# Transformer itself
class TransformerModel(nn.Module):
    def __init__(self, vocab_size, block_size, vector_size=128, num_heads=4, num_blocks=4, dropout=0.1):
        super().__init__()
        self.block_size = block_size
        self.token_embeddings = nn.Embedding(vocab_size, vector_size) # learned vector for each token
        self.position_embeddings = nn.Embedding(block_size, vector_size) # learned vector for each position because attention is blind to order
        self.blocks = nn.Sequential(*[Block(vector_size, num_heads, block_size, dropout) for _ in range(num_blocks)]) # Put blocks in order
        self.ln_f = nn.LayerNorm(vector_size) # Normalize
        self.head = nn.Linear(vector_size, vocab_size) # back to scores over all chars

    def forward(self, x, y=None):
        batch, seq_len = x.shape
        hidden = self.token_embeddings(x) + self.position_embeddings(torch.arange(seq_len, device=x.device)) # token representation + positional representation
        hidden = self.blocks(hidden) # gather context + detect patterns through each block
        logits = self.head(self.ln_f(hidden)) # normalize then score every next character
        if y is None: # generating so no loss
            return logits, None
        loss = F.cross_entropy(logits.view(batch * seq_len, -1), y.view(batch * seq_len)) # Compute loss (-log(probability of right answer))
        return logits, loss

    @torch.no_grad()
    def generate(self, encoded_prompt, max_new_tokens):
        for _ in range(max_new_tokens):
            relevant_part_of_prompt = encoded_prompt[:, -self.block_size:] # Can only percieve block_size tokens at a time so take last 128 from prompt
            logits, _ = self(relevant_part_of_prompt) # Next char scores at every position
            probs = F.softmax(logits[:, -1, :], dim=-1) # Take scores at the last position for next char, convert to probabilities
            next_id = torch.multinomial(probs, 1) # Randomly sample weighted by probability to avoid repetition
            encoded_prompt = torch.cat([encoded_prompt, next_id], dim=1) # append the new token, so the next loop sees it.
        return encoded_prompt