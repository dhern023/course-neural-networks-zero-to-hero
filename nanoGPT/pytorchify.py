"""
Uses the full torch API

Self-Attention:
    Operates on the target sequence. query, key, and value are all derived from the target sequence.
Cross-Attention: Operates between the target sequence and the encoded memory.
    query comes from the target sequence. key and value come from the encoded memory.

BUG: (?) Removing the memory_mask parameter dramatically reduces the loss.
"""
import matplotlib.pyplot as plt
import pathlib
import seaborn
import torch
import torch.nn
import torch.utils.data


# in house
import _tokenizer
import models_pytorchify

# Defaults
DIRNAME_OUT = "nanoGPT"
DIR_READ = pathlib.Path(__file__).resolve().parent
DIR_OUT = pathlib.Path(__file__).resolve().parents[1] / "out" / DIRNAME_OUT
DIR_OUT.mkdir(exist_ok=True, parents=True)

FNAME_DATA = DIR_READ / "input.txt"

g = torch.Generator().manual_seed(2147483647) # default seed from the torch docs, shares memory

if FNAME_DATA.exists():
    with open(FNAME_DATA, "r") as file:
        data = file.read()
else:
    sys.exit(9) # fail fast

dict_to_idx, dict_to_token = _tokenizer.construct_character_mappings(data)
list_tokens = _tokenizer.character_encode(data, dict_to_idx)
vector_tokens = torch.tensor(list_tokens, dtype = torch.long)

SIZE_CONTEXT=8 # 256
SIZE_BATCH=4 # 64
SIZE_VOCAB=len(dict_to_idx)
SIZE_EMBEDDING_DIM=32 # 384
LEARNING_RATE = 1e-5 # 4e-3

# pad to account for missing data
remainder = len(vector_tokens) % (SIZE_CONTEXT+1)
size_pad = (SIZE_CONTEXT+1) - remainder
vector_tokens = torch.cat((vector_tokens, torch.zeros(size_pad, dtype=vector_tokens.dtype)))
matrix_tokens = vector_tokens.view(-1, SIZE_CONTEXT+1)

# split train/test

dataset = torch.utils.data.TensorDataset(
    matrix_tokens[:,:SIZE_CONTEXT], # first SIZE_CONTEXT
    matrix_tokens[:,1:SIZE_CONTEXT+1] # shift by one to get targets
)
dataset_train, dataset_test = torch.utils.data.random_split(dataset, [0.9, 0.1], generator=g) # 90/10

dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=SIZE_BATCH, shuffle=True)
dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=SIZE_BATCH, shuffle=False)

# Train model -------------------------------------------------------------------------------------

class CharacterLevelAutoregressor(torch.nn.Module):
    """
    Treat logits = C[X]
    Then use attention (multi)

    Informed it's much faster to split the decoder and transformer separately for parallel reasons
    We need to provide the mask ourselves

    """
    def __init__(self, size_context, num_embeddings, size_embedding, num_heads, num_blocks):
        super().__init__()
        scalar = 4 # hard-coded
        self.embedding_tokens = torch.nn.Embedding(num_embeddings=num_embeddings, embedding_dim=size_embedding)
        self.embedding_token_position = torch.nn.Embedding(num_embeddings=size_context, embedding_dim=size_embedding)
        self.layer_decoder = torch.nn.TransformerDecoderLayer(
            d_model=size_embedding, nhead=num_heads, dim_feedforward=scalar*size_embedding, dropout=0.2, bias=False, norm_first=True
        )
        self.transformer_decoder = torch.nn.TransformerDecoder(decoder_layer=self.layer_decoder, num_layers=num_blocks)
        self.projection_decoder = torch.nn.Linear(size_embedding, num_embeddings) # (size_head, num_embeddings)

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

        tgt_mask = torch.nn.Transformer.generate_square_subsequent_mask(sz=T)
        if self.layer_decoder.self_attn.batch_first: # inputs are size (B, T, size_embedding)
            blocks = self.transformer_decoder(tgt=input_embeddings, memory=input_embeddings, tgt_mask=tgt_mask, memory_mask=tgt_mask)
        else: # inputs need to be size (T, B, size_embedding)
            input_embeddings = input_embeddings.permute(1, 0, 2) # (T, B, size_embedding)
            blocks = self.transformer_decoder(tgt=input_embeddings, memory=input_embeddings, tgt_mask=tgt_mask, memory_mask=tgt_mask)
            blocks = blocks.permute(1, 0, 2) # (B, T, size_embedding)
        logits = self.projection_decoder(blocks) # (B, T, size_embedding)  * (size_embedding, num_embeddings)^T -> (B, T, num_embeddings)

        loss = None

        if targets is not None:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)

            targets = targets.view(B*T)
            loss = torch.nn.functional.cross_entropy(logits, targets)

        return logits, loss

    @torch.no_grad()
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

# Train model -------------------------------------------------------------------------------------
NUM_HEADS = 4 # 6
NUM_BLOCKS = 4
fname_model = 'character_model_state_dict.pth'
fname_hook = 'character_model_hook.pth'

# model = models_pytorchify.BigramLanguageModelAttentionPytorchify(SIZE_CONTEXT, SIZE_VOCAB, SIZE_EMBEDDING_DIM, NUM_HEADS, NUM_BLOCKS)
model = CharacterLevelAutoregressor(SIZE_CONTEXT, SIZE_VOCAB, SIZE_EMBEDDING_DIM, NUM_HEADS, NUM_BLOCKS)
if (DIR_OUT / fname_model).exists():
    model.load_state_dict(torch.load(DIR_OUT / fname_model))
else: # need to train

    SIZE_EVALUATE=1000

    @torch.no_grad()
    def evaluate_model(instance_model):
        dict_out = {}
        instance_model.eval()
        for (key, dataloader) in zip(("train", "test"), (dataloader_train, dataloader_test)):
            tensor_losses = torch.zeros(SIZE_EVALUATE)
            for i, batch in enumerate(dataloader):
                xs, ys = batch
                logits, loss = instance_model(xs, ys)
                tensor_losses[i] = loss.item()
                if i == SIZE_EVALUATE-1:
                    break
            dict_out[key] = tensor_losses.mean()
        instance_model.train()

        return dict_out

    # Hooks to store the activations
    dict_activations = {}
    def get_activation(name):
        # the hook signature
        def hook(model, input, output):
            dict_activations.setdefault(name, [])
            if isinstance(output, tuple):
                dict_activations[name].append(output[0].detach())
            else:
                dict_activations[name].append(output.detach())
        return hook

    dict_gradients = {}
    def get_gradient(name):
        # the hook signature
        def hook(module, grad_input, grad_output):
            dict_gradients.setdefault(name, [])
            dict_gradients[name].append(grad_output[0])
        return hook

    # register hooks on all layers
    list_hooks = []
    for name, module in model.named_modules():
        list_hooks.append(module.register_forward_hook(get_activation(name))) 
        list_hooks.append(module.register_full_backward_hook(get_gradient(name)))

    # train model
    optimizer = torch.optim.AdamW(params = model.parameters(), lr = LEARNING_RATE)
    NUM_EPOCHS=1 # runs through the dataset
    for epoch in range(NUM_EPOCHS):
        for i, batch in enumerate(dataloader_train):

            if i % SIZE_EVALUATE == 0:
                dict_evaluate = evaluate_model(model)
                print("iteration", i, "train_loss", dict_evaluate["train"], "test_loss", dict_evaluate["test"])

            xtr, ytr = batch
            optimizer.zero_grad()

            log_probs, loss = model(xtr, ytr)

            loss.backward()
            optimizer.step()

            # detach the hooks to only save after the first run
            for hook in list_hooks:
                hook.remove()

        print(loss)

    dict_out = {
        "activations": dict_activations,
        "gradients": dict_activations
    }

    torch.save(model.state_dict(), DIR_OUT / fname_model)
    torch.save(dict_out, DIR_OUT / fname_hook)

# Evaluate ----------------------------------------------------------------------------------------

def plot_attention_weights(attention_weights, layer_idx):
    """
    Plot attention weights for a given transformer layer
    attention_weights: Tensor of shape (num_heads, seq_len, seq_len)
    layer_idx: index of the layer for which attention is being plotted
    """
    num_heads = attention_weights.shape[0]
    fig, axes = plt.subplots(1, num_heads, figsize=(20, 5))

    for head_idx in range(num_heads):
        ax = axes[head_idx]
        attention_map = attention_weights[head_idx].cpu().detach().numpy()
        seaborn.heatmap(attention_map, ax=ax, cmap='viridis', cbar=False)
        ax.set_title(f"Layer {layer_idx} - Head {head_idx}")

    plt.show()

def plot_attention_weights(attention_weights, layer_idx, head_idx):
    """
    Plot attention weights for a given transformer layer and head
    attention_weights: Tensor of shape (num_layers, num_heads, seq_len, seq_len)
    layer_idx: index of the layer for which attention is being plotted
    head_idx: index of the attention head for which attention is being plotted
    """
    attention_map = attention_weights[layer_idx][head_idx].cpu().detach().numpy()
    plt.figure(figsize=(8, 6))
    sns.heatmap(attention_map, cmap='viridis', cbar=True)
    plt.title(f"Layer {layer_idx} - Head {head_idx} Attention")
    plt.xlabel("Target Token Position")
    plt.ylabel("Source Token Position")
    out = "attention_weights.png"
    plt.savefig(DIR_OUT / out)


def visualize_attention_weights(input, model):
    """
    Not compatible with BigramLanguageModelAttentionPytorchify
        due to assuming model has a transfomer_decoder layer
    """
    model.eval()  # Switch to evaluation mode
    with torch.no_grad():
        tokens = model.embedding_tokens(input)
        positions = model.embedding_token_position(torch.arange(input.shape[1]))
        input_embeddings = tokens + positions

        tgt_mask = torch.nn.Transformer.generate_square_subsequent_mask(sz=input.shape[1])
        if model.layer_decoder.self_attn.batch_first:
            blocks = model.transformer_decoder(tgt=input_embeddings, memory=input_embeddings, tgt_mask=tgt_mask)
        else:
            input_embeddings = input_embeddings.permute(1, 0, 2)
            blocks = model.transformer_decoder(tgt=input_embeddings, memory=input_embeddings, tgt_mask=tgt_mask)
            blocks = blocks.permute(1, 0, 2)

        attention_weights = model.layer_decoder.self_attn.in_proj_weight.cpu().numpy()
        num_heads = attention_weights.shape[1]
        fig, axs = plt.subplots(nrows=1, ncols=num_heads, figsize=(15, 5))

        index_layer=0
        index_head=0
        attention_map = attention_weights
        plt.figure(figsize=(8, 6))
        seaborn.heatmap(attention_map[:attention_weights.shape[0]//4,:attention_weights.shape[1]//4], cmap='viridis', cbar=True)
        plt.title(f"Layer {index_layer} - Head {index_head} Attention")
        plt.xlabel("Target Token Position")
        plt.ylabel("Source Token Position")

        out = "attention_weights.png"
        plt.savefig(DIR_OUT / out)

tokens_input = torch.tensor(_tokenizer.character_encode("First", dict_to_idx), dtype = torch.long).view(-1,1)
visualize_attention_weights(tokens_input, model)

# Sample ------------------------------------------------------------------------------------------
tokens_single_char = torch.zeros((1,1), dtype = torch.long)
sample = model.generate(tokens_single_char, SIZE_CONTEXT, 500) # 2d tensor
sample_output = _tokenizer.character_decode(sample[0].tolist(), dict_to_token)
print(sample_output)
