import torch

class BigramLanguageModel(torch.nn.Module):
    """
    Treat logits = C[X]
    Then P = softmax(C[X])

    Weak since tokens don't talk to each other
    """
    def __init__(self, size_vocab):
        super().__init__()
        self.C = torch.nn.Embedding(num_embeddings=size_vocab, embedding_dim=size_vocab)

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
            logits = logits[:,-1,:] # last token in each sequence reduce to (batch_size, size_vocab)
            P = torch.nn.functional.softmax(logits, dim=-1) # exp(log_counts) / row_sum = P
            tensor_out_next = torch.multinomial(P, num_samples=1, replacement=True)
            tensor_out = torch.cat([tensor_out, tensor_out_next], dim = 1)
        
        return tensor_out


