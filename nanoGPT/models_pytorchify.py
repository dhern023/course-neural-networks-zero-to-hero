"""
Replaced certain portions with their pytorch equivalents to streamline the process.
A means to gain insight how torch.nn.TransformerDecoderLayer works
    Multiheadattn_mask -> tgt_mask
    But then 

Andrej uses n_embed as the size_dimension, which can be confusing.
His block size is size_context
B = size_batch
T = time and size_context
C = channels and size_embedding
"""
import torch

class FeedForward(torch.nn.Module):
    """
    max(0, xW1 + b1)W2 + b2
    = (ReLu(xW1+B1))W2 + b2

    The projection is needed to go back into the residual pathway
    """
    def __init__(self, size_embedding):
        """
        Hard-coded to scale the inner layer by 4 per the paper.
        """
        super().__init__()
        self.layer = torch.nn.Linear(in_features=size_embedding, out_features=4*size_embedding)
        self.relu = torch.nn.ReLU()
        self.projection = torch.nn.Linear(in_features=4*size_embedding, out_features=size_embedding)
        self.dropout = torch.nn.Dropout(p=0.2)

    def forward(self, input):
        """
        input shape (size_context, size_embedding)
        """
        layer = self.layer(input) # (size_context, size_embedding) * (size_embedding, size_embedding).T
        activation = self.relu(layer)
        projection = self.projection(activation)
        out = self.dropout(projection)

        return out

class Block(torch.nn.Module):

    def __init__(self, size_context, size_embedding, num_heads):
        super().__init__()

        self.attention_heads = torch.nn.MultiheadAttention(embed_dim=size_embedding, num_heads=num_heads, dropout=0.2, batch_first=True)
        self.projection = torch.nn.Linear(in_features=size_embedding, out_features=size_embedding)
        self.feed_forward = FeedForward(size_embedding)
        self.prenorm_attention = torch.nn.LayerNorm(size_embedding)
        self.postnorm_attention = torch.nn.LayerNorm(size_embedding)
        self.postnorm_feedforward = torch.nn.LayerNorm(size_embedding)

    def forward(self, input):
        """
        input is of size (B, size_context, size_embedding)

        Adds the residual forks via addition
        The projection is needed to go back into the residual pathway
        """
        B, T, C = input.shape
        input_normed = self.prenorm_attention(input)
        tgt_mask = torch.nn.Transformer.generate_square_subsequent_mask(sz=T)
        attention_output, attention_weights = self.attention_heads(query=input_normed, key=input_normed, value=input_normed, attn_mask=tgt_mask)
        attention = input_normed + self.projection(attention_output) # (B, T, size_embedding)
        attention_normed = self.postnorm_attention(attention)
        feed_forward = attention_normed + self.feed_forward(attention_normed) # (B, T, size_embedding)
        out = self.postnorm_feedforward(feed_forward)

        return out

class BigramLanguageModelAttentionPytorchify(torch.nn.Module):
    """
    Treat logits = C[X]
    Then use attention (multi)
    """
    def __init__(self, size_context, num_embeddings, size_embedding, num_heads, num_blocks):
        super().__init__()
        self.embedding_tokens = torch.nn.Embedding(num_embeddings=num_embeddings, embedding_dim=size_embedding)
        self.embedding_token_position = torch.nn.Embedding(num_embeddings=size_context, embedding_dim=size_embedding)
        self.blocks = torch.nn.Sequential(
            *(Block(size_context, size_embedding, num_heads) for i in range(num_blocks))
        )
        self.projection_decoder = torch.nn.Linear(size_embedding, num_embeddings) # (size_head, num_embeddings)

        # Share weights between attention head and output projection
        self.blocks.attention_heads.in_proj_weight = self.projection_decoder.weight

    def forward(self, input, targets):
        """
        Inputs/Targets are of shape (size_batch, size_context)
        Reshape things accordingly for torch.nn.functional.cross_entropy(., C, ...)

        cross_entropy = NLL(softmax( logits ))
        NOTE: Remember, pytorch calls matmul() on the transpose
        """
        B, T = input.shape
        tokens = self.embedding_tokens(input) # (input.shape(), Channels) = (B, T, size_embedding)
        positions = self.embedding_token_position(torch.arange(T)) # (input.shape(), Channels) = (T, size_embedding)
        input_embeddings = tokens + positions # broadcasted to (B, T, size_embedding)
        blocks = self.blocks(input_embeddings) # (B, T, size_embedding)
        logits = self.projection_decoder(blocks) # (B, T, size_embedding)  * (size_embedding, num_embeddings)^T -> (B, T, num_embeddings)

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
            context = tensor_out[:, -size_context:]
            logits, loss = self(context, None) # log counts
            logits = logits[:,-1,:] # last token in each sequence reduce to (batch_size, num_embeddings)
            P = torch.nn.functional.softmax(logits, dim=-1) # exp(log_counts) / row_sum = P
            tensor_out_next = torch.multinomial(P, num_samples=1, replacement=True)
            tensor_out = torch.cat([tensor_out, tensor_out_next], dim = 1)

        return tensor_out
