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
        logits = self.C(input) # (input.shape(), Channels) = (size_batch, size_context, embedding_dim)
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
        # Explicitly torch.einsum('btm,mib -> bti',tensor_query, tensor_key.T) # (B, T, m) * (B, T, m)^T = (B, T, T)
        tensor_wei = torch.einsum('btm,bim -> bti',tensor_query, tensor_key) # (B, T, T)
        tensor_wei = tensor_wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T) with no future context
        # calculate attention
        tensor_wei_probs = torch.nn.functional.softmax(tensor_wei, dim = -1) # (B, T, T)
        out = torch.einsum('btj,bjm->btm', tensor_wei_probs, tensor_value) # (B, T, m)

        return out


