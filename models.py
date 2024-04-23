import torch
import torch.nn as nn
import torch.nn.functional as F

class MySelfAttention(nn.Module):
    """
    Self attention layer
    """
    def __init__(self, input_dim):
        """
        :param input_dim: The feature dimension the input tokens (d).
        """
        super(MySelfAttention, self).__init__()
        self.input_dim = input_dim
        ### YOUR CODE HERE ###
        # make the metrix of W_q, W_k, W_v
        self.W_q = nn.Linear(input_dim, input_dim)
        self.W_k = nn.Linear(input_dim, input_dim)
        self.W_v = nn.Linear(input_dim, input_dim)


    def forward(self, x):
        ### YOUR CODE HERE ###
        # calculate Q, K, V
        Q = self.W_q(x) # (batch_size, max_len, input_dim)
        K = self.W_k(x) # (batch_size, max_len, input_dim)
        V = self.W_v(x) # (batch_size, max_len, input_dim)

        # calculate attention score
        attention = torch.bmm(Q, K.transpose(1,2)) / (self.input_dim ** 0.5) # bmm : batch matrix multiplication
        attention = F.softmax(attention, dim=2) 
        values = torch.bmm(attention, V)
        return values
    
class MyLayerNorm(nn.Module):
    """
    Layer Normalization layer.
    """
    def __init__(self, input_dim):
        """
        :param input_dim: The dimension of the input (T, d).
        """
        super(MyLayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(*input_dim))
        self.beta = nn.Parameter(torch.zeros(*input_dim))

    def forward(self, x):
        ### YOUR CODE HERE ###
        mean = x.mean(dim=(1,2), keepdim=True) # (batch_size, 1, 1)
        std = x.std(dim=(1,2), keepdim=True) # (batch_size, 1, 1)
        return self.gamma * (x - mean) / (std + 1e-8) + self.beta
    
class MyTransformerBlock(nn.Module):
    """
    Transformer block.
    """
    def __init__(self, max_len, input_dim):
        super(MyTransformerBlock, self).__init__()
        self.attention = MySelfAttention(input_dim)
        self.norm1 = MyLayerNorm((max_len, input_dim))
        self.norm2 = MyLayerNorm((max_len, input_dim))
        self.fc1 = nn.Linear(input_dim, input_dim)
        self.fc2 = nn.Linear(input_dim, input_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        out = self.attention(x)
        x = self.norm1(self.dropout(out) + x)
        out = self.fc2(F.relu(self.fc1(x)))
        out = self.norm2(out + x)
        return out

class MyTransformer(nn.Module):
    """
    Transformer.
    """
    def __init__(self, vocab, max_len, num_of_blocks):
        """
        :param vocab: The vocabulary object.
        :param num_of_blocks: The number of transformer blocks.
        """
        super(MyTransformer, self).__init__()
        self.embedding = nn.Embedding.from_pretrained(vocab.vectors)
        self.emb_dim = self.embedding.embedding_dim
        self.max_len = max_len
        self.blocks = nn.ModuleList([MyTransformerBlock(self.max_len, self.emb_dim) for _ in range(num_of_blocks)])
        self.fc = nn.Linear(self.emb_dim, 1)

    def forward(self, x):
        x = self.embedding(x)
        for block in self.blocks:
            x = block(x)
        avg_pooling = x.mean(dim=1)
        x = self.fc(avg_pooling)
        return x

