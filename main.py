# We continue the work started in main.ipynb. We report here the code that we have written in the notebook and we actually need.

import torch
import torch.nn as nn
from torch.nn import functional as F

# Hyperparameters
batch_size = 32 # number of independent sequences that are trained in parallel
block_size = 8 # We define the context window size, also called block size
max_epochs = 5000
eval_interval = 500
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
embedding_size = 32
# ------------

torch.manual_seed(1337)

# wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# here are all the unique characters that occur in this text
chars = sorted(list(set(text)))
vocab_size = len(chars)
# we will create a mapping from characters to indices and vice versa. This is also known as tokenization.
char_to_idx = {ch:i for i,ch in enumerate(chars)}
idx_to_char = {i:ch for i,ch in enumerate(chars)}
# Then we make to lambda functions to convert strings to list of indices and vice versa
encode = lambda x: [char_to_idx[ch] for ch in x]
decode = lambda x: ''.join([idx_to_char[i] for i in x])

# Train and test splits
data = torch.tensor(encode(text), dtype=torch.long)
# Now we split the dataset into validation and training sets
train_size = int(0.9 * len(data))
val_data = data[train_size:]
train_data = data[:train_size]

# Now we make a function that generates a batch of training data or validation data based on the input
def get_batch(split):
    if split == 'train':
        data = train_data
    else:
        data = val_data
    # We want to generate a batch of random sequences, to do this we will randomly sample a starting index for each sequence.
    # This index has to be between 0 and the length of the dataset minus the block size.
    ix = torch.randint(0, data.size(0) - block_size, (batch_size,))
    # Each block is identified by an x and y (label) tensor
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix]) # y will be the same as x, except shifted one position to the right
    x, y = x.to(device), y.to(device) # move the data to the device (GPU or CPU)
    return x, y

@torch.no_grad()
def estimate_loss():
    # This function estimates the loss of the model on the train and val sets as an average of the loss over multiple epochs.
    out = {}
    model.eval()  # Set the model to evaluation mode (some nn layers may behave differently in train and eval mode)
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)  # Tensor to store loss values
        for k in range(eval_iters):
            X, Y = get_batch(split)  # Get a batch of data
            logits, loss = model(X, Y)  # Forward pass
            losses[k] = loss.item()  # Store the loss value
        out[split] = losses.mean()  # Compute the mean loss for the split
    model.train()  # Set the model back to training mode
    return out

class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        # for each token of the vocabulary, we will have a corresponding embedding. The right value for the embedding are learned 
        # during training. The torch method Embedding is used to create the embedding table. This is of size vocab_size x embedding_size, 
        # where embedding_size is a hyperparameter that we can choose.
        # Remember that the embedding of a token is the starting point to make the predictions.
        self.token_embedding_table = nn.Embedding(vocab_size, embedding_size)
        # If later we want to introduce an attention block it is also important the position of each token. For this reason, we will
        # also need to add the position embedding to the token embedding.
        self.position_embedding_table = nn.Embedding(block_size, embedding_size) # Notice the block_size here.
        # We define a linear layer that is used to go from the token embeddings to the logits. 
        self.lm_head = nn.Linear(embedding_size, vocab_size)
        
    def forward(self, idx, target=None):
        # idx: A tensor of shape (batch_size, block_size) containing token IDs.
        # - batch_size: Number of sequences (blocks) processed in parallel.
        # - block_size: Number of tokens in each sequence (i.e., the length of each block).
        # Each element in idx represents a token in the vocabulary, serving as input to the model
        # to retrieve embeddings and predict the probabilities of the next tokens.
        batch_size, block_size = idx.shape
        
        token_embedding = self.token_embedding_table(idx) # shape of token_embedding: (batch_size, block_size, embedding_size)
        # the input for the position embedding is the position of the all the tokens in the sequence
        position_embedding = self.position_embedding_table(torch.arange(block_size, device=device)) # shape: (block_size, embedding_size)
        
        x = token_embedding + position_embedding # shape: (batch_size, block_size, embedding_size)
        
        logits =  self.lm_head(x) # shape: (batch_size, block_size, vocab_size) 
        
        
        if target == None:
            loss = None
        else:
            # To be compliant with the cross-entropy loss, we need to reshape the logits tensor to (batch_size * block_size, embedding_size).
            # Remember: embedding_size == vocab_size
            logits = logits.view(-1, vocab_size)
            target = target.view(-1)
            
            # The loss tells us how well the model is doing on the training data. 
            loss = F.cross_entropy(logits, target)
        
        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        
        # NOTE: Since we are using a Bigram model, we will only consider the last token in the sequence to predict the next token.
        # This function though, is constructed in a more general way, so in the future we can also exploit the other tokens in the sequence. 
        
        # like in the forward method, idx is a tensor of shape (batch_size, block_size) containing token IDs.
        for t in range(max_new_tokens):
            # We get the predictions for the next token, i.e. the logits
            logits, _ = self.forward(idx)
            # But to predict the next token, we only need the last prediction, which is the one at the last position of the sequence.
            last_logit = logits[:, -1, :] # shape: (batch_size, embedding_size)
            # We get the token with the highest probability applying the softmax function to the last prediction
            probabilities = F.softmax(last_logit, dim=-1) # shape: (batch_size, embedding_size)
            # Sample from the probability distribution to get the token ID of the next token
            next_token = torch.multinomial(probabilities, num_samples=1) # shape: (batch_size, 1)
            # We add the new token to the sequence
            idx = torch.cat([idx, next_token], dim=1) # shape: (batch_size, block_size + 1)
            
        return idx
            
model = BigramLanguageModel(vocab_size)
# if we are using a GPU, we need to move the model to the GPU
m = model.to(device)

# Now we will train the model. 
# We will use the Adam W optimizer, which is a popular optimizer for training neural networks.
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
# The optimizer takes care of updating the model's parameters based on the computed gradients.

# Training loop
for epoch in range(max_epochs):
    # every once in a while evaluate the loss on train and val sets
    if epoch % eval_interval == 0 or epoch == max_epochs - 1:
        losses = estimate_loss()
        print(f"step {epoch}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
    # Get a batch of training data
    x_batch, y_batch = get_batch('train')
    # Evaluate the model on the batch using the loss
    _, loss = model(x_batch, y_batch)
    # Reset the gradients to zero
    optimizer.zero_grad(set_to_none=True)
    # Compute the gradients for the parameters
    loss.backward()
    # Update the parameters of the model
    optimizer.step()

# generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))