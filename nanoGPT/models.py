"""
Andrej uses n_embed as the size_dimension, which can be confusing
"""
import torch

class BigramLanguageModel(torch.nn.Module):
    """
    Treat logits = C[X] (num_embeddings needed, space for each embedding)
    Then P = softmax(C[X])

    Weak since tokens don't talk to each other
    """
    def __init__(self, num_embeddings, size_embedding):
        """
        Need one embedding for each possible token
        Size of each embedding is a hyper-parameter
        """
        super().__init__()
        self.C = torch.nn.Embedding(num_embeddings=num_embeddings, embedding_dim=size_embedding)

    def forward(self, input, targets):
        """
        Inputs/Targets are of shape (size_batch, size_context)
        Reshape things accordingly for torch.nn.functional.cross_entropy(., C, ...)

        cross_entropy = NLL(softmax( logits ))
        """
        logits = self.C(input) # (input.shape(), Channels) = (size_batch, size_context, size_embedding)
        loss = None

        if targets is not None:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)

            targets = targets.view(B*T)
            loss = torch.nn.functional.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, input, max_new_tokens):
        tensor_out = input.clone()
        for i in range(max_new_tokens):
            logits, loss = self(tensor_out, None) # log counts
            logits = logits[:,-1,:] # last token in each sequence reduce to (batch_size, num_embeddings)
            P = torch.nn.functional.softmax(logits, dim=-1) # exp(log_counts) / row_sum = P
            tensor_out_next = torch.multinomial(P, num_samples=1, replacement=True)
            tensor_out = torch.cat([tensor_out, tensor_out_next], dim = 1)

        return tensor_out

class Head(torch.nn.Module):
    """
    Implements attention = softmax((XW_q)(XW_k)^T)(XW_v)
    """
    def __init__(self, size_context, size_embedding, size_head):
        super().__init__()
        self.size_head = size_head
        self.attention_query = torch.nn.Linear(size_embedding, size_head, bias=False)
        self.attention_key = torch.nn.Linear(size_embedding, size_head, bias=False)
        self.attention_value = torch.nn.Linear(size_embedding, size_head, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(size_context, size_context))) # not a parameter


    def forward(self, input):
        """
        Inputs/Targets are of shape (size_batch, size_context, size_embedding)
        For each batch:
            Treat input as C[X]
            Linearly project into Q, K, V
            Calculate W = QK^T
            Mask it to lower triangular to make it autoregressive (no-future)
            Finally, softmax(W)*V
        """
        B, T, C = input.shape
        tensor_query = self.attention_query(input) # (size_batch, size_context, size_head)
        tensor_key = self.attention_key(input) # (size_batch, size_context, size_head)
        tensor_value = self.attention_value(input) # (size_batch, size_context, size_head)
        # calculate attention scores
        scalar = self.size_head ** -0.5
        # Explicitly torch.einsum('btd,dib -> bti',tensor_query, tensor_key.T) # (B, T, d) * (B, T, d)^T = (B, T, T)
        tensor_wei = torch.einsum('btd,bid -> bti',tensor_query, tensor_key) # (B, T, T)
        tensor_wei = tensor_wei * scalar
        tensor_wei = tensor_wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T) with no future context
        # calculate attention
        tensor_wei_probs = torch.nn.functional.softmax(tensor_wei, dim = -1) # (B, T, T)
        out = torch.einsum('btj,bjd->btd', tensor_wei_probs, tensor_value) # (B, T, d)

        return out

class BigramLanguageModelAttention(torch.nn.Module):
    """
    Treat logits = C[X]
    Then use attention
    """
    def __init__(self, size_context, num_embeddings, size_embedding, size_head):
        super().__init__()
        self.embedding_tokens = torch.nn.Embedding(num_embeddings=num_embeddings, embedding_dim=size_embedding)
        self.embedding_token_position = torch.nn.Embedding(num_embeddings=size_context, embedding_dim=size_embedding)
        self.self_attention_head = Head(size_context, size_embedding, size_head)
        self.projection_decoder = torch.nn.Linear(size_head, num_embeddings) # (size_head, num_embeddings)

    def forward(self, input, targets):
        """
        Inputs/Targets are of shape (size_batch, size_context)
        Reshape things accordingly for torch.nn.functional.cross_entropy(., C, ...)

        cross_entropy = NLL(softmax( logits ))
        NOTE: Remember, pytorch calls matmul()
        """
        B, T = input.shape
        tokens = self.embedding_tokens(input) # (input.shape(), Channels) = (B, T, size_embedding)
        positions = self.embedding_token_position(torch.arange(T)) # (input.shape(), Channels) = (T, size_embedding)
        input_embeddings = tokens + positions # broadcasted to (B, T, size_embedding)
        attention = self.self_attention_head(input_embeddings) # (B, T, d)
        logits = self.projection_decoder(attention) # (B, T, d)  * (d, num_embeddings)^T -> (B, T, num_embeddings)

        loss = None

        if targets is not None:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)

            targets = targets.view(B*T)
            loss = torch.nn.functional.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, input, size_context, max_new_tokens):
        """
        input is passed to (T, size_embedding) and (T, size_embedding)
        """
        tensor_out = input.clone()
        for i in range(max_new_tokens):
            tensor_input = tensor_out[:, -size_context:]
            logits, loss = self(tensor_input, None) # log counts
            logits = logits[:,-1,:] # last token in each sequence reduce to (batch_size, num_embeddings)
            P = torch.nn.functional.softmax(logits, dim=-1) # exp(log_counts) / row_sum = P
            tensor_out_next = torch.multinomial(P, num_samples=1, replacement=True)
            tensor_out = torch.cat([tensor_out, tensor_out_next], dim = 1)

        return tensor_out

class MultiHeadAttention(torch.nn.Module):
    """
    Implementation that gives uniform head size for each head s.t.,
        size_head = size_embedding // num_heads
    """
    def __init__(self, size_context, size_embedding, num_heads):
        """
        """
        super().__init__()
        self.size_embedding = size_embedding
        self.size_head = size_embedding // num_heads
        self.heads = torch.nn.ModuleList([Head(size_context, size_embedding, self.size_head) for i in range (num_heads)])

    def forward(self, input):
        """
        Concatentate the across the communication channel dimension (size_head) since num_heads * size_head = size_embedding
        NOTE: Don't like using -1 as a dimension, but there must be a safe reason for it.
        """
        return torch.cat([head(input) for head in self.heads], dim=-1) # (B, T, size_head * num_heads)

class BigramLanguageModelAttentionMulti(torch.nn.Module):
    """
    Treat logits = C[X]
    Then use attention (multi)
    """
    def __init__(self, size_context, num_embeddings, size_embedding, num_heads):
        super().__init__()
        self.embedding_tokens = torch.nn.Embedding(num_embeddings=num_embeddings, embedding_dim=size_embedding)
        self.embedding_token_position = torch.nn.Embedding(num_embeddings=size_context, embedding_dim=size_embedding)
        self.self_attention_heads = MultiHeadAttention(size_context, size_embedding, num_heads)
        self.projection_decoder = torch.nn.Linear(size_embedding, num_embeddings) # (size_head, num_embeddings)

    def forward(self, input, targets):
        """
        Inputs/Targets are of shape (size_batch, size_context)
        Reshape things accordingly for torch.nn.functional.cross_entropy(., C, ...)

        cross_entropy = NLL(softmax( logits ))
        NOTE: Remember, pytorch calls matmul()
        """
        B, T = input.shape
        tokens = self.embedding_tokens(input) # (input.shape(), Channels) = (B, T, size_embedding)
        positions = self.embedding_token_position(torch.arange(T)) # (input.shape(), Channels) = (T, size_embedding)
        input_embeddings = tokens + positions # broadcasted to (B, T, size_embedding)
        # size_head = size_embeddings // num_heads
        attention = self.self_attention_heads(input_embeddings) # (B, T, d * num_heads) = (B, T, size_embedding)
        logits = self.projection_decoder(attention) # (B, T, size_embedding)  * (size_embedding, num_embeddings)^T -> (B, T, num_embeddings)

        loss = None

        if targets is not None:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)

            targets = targets.view(B*T)
            loss = torch.nn.functional.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, input, size_context, max_new_tokens):
        """
        input is passed to (T, size_embedding) and (T, size_embedding)
        """
        tensor_out = input.clone()
        for i in range(max_new_tokens):
            tensor_input = tensor_out[:, -size_context:]
            logits, loss = self(tensor_input, None) # log counts
            logits = logits[:,-1,:] # last token in each sequence reduce to (batch_size, num_embeddings)
            P = torch.nn.functional.softmax(logits, dim=-1) # exp(log_counts) / row_sum = P
            tensor_out_next = torch.multinomial(P, num_samples=1, replacement=True)
            tensor_out = torch.cat([tensor_out, tensor_out_next], dim = 1)

        return tensor_out
