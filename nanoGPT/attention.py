"""
Trains an attention bigram model on the input text
"""
import pathlib
import torch
import torch.utils.data
import sys

# in house
import _tokenizer
import models

FNAME_DATA = pathlib.Path(__file__).parent / "input.txt"

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
LEARNING_RATE = 1e-3 # 4e-3

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
SIZE_HEAD = 16
NUM_HEADS = 4 # 6
NUM_BLOCKS = 4

model_name = "residual"
if model_name == "single":
    model = models.BigramLanguageModelAttention(SIZE_CONTEXT, SIZE_VOCAB, SIZE_EMBEDDING_DIM, SIZE_HEAD)
elif model_name == "multi":
    model = models.BigramLanguageModelAttentionMulti(SIZE_CONTEXT, SIZE_VOCAB, SIZE_EMBEDDING_DIM, NUM_HEADS)
elif model_name == "forward":
    model = models.BigramLanguageModelAttentionMultiForward(SIZE_CONTEXT, SIZE_VOCAB, SIZE_EMBEDDING_DIM, NUM_HEADS)
elif model_name == "residual":
    model = models.BigramLanguageModelAttentionComplete(SIZE_CONTEXT, SIZE_VOCAB, SIZE_EMBEDDING_DIM, NUM_HEADS, NUM_BLOCKS)


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

# train model
optimizer = torch.optim.AdamW(params = model.parameters(), lr = LEARNING_RATE)
NUM_EPOCHS=1 # runs through the dataset
for epoch in range(NUM_EPOCHS):
    for i, batch in enumerate(dataloader_train):

        if i % SIZE_EVALUATE == 0:
            dict_evaluate = evaluate_model(model)
            print("iteration", i, "train_loss", dict_evaluate["train"], "test_loss", dict_evaluate["test"])

        xtr, ytr = batch
        model.zero_grad()

        log_probs, loss = model(xtr, ytr)

        loss.backward()
        optimizer.step()
    print(loss)

tokens_single_char = torch.zeros((1,1), dtype = torch.long)
sample = model.generate(input = tokens_single_char, size_context=SIZE_CONTEXT, max_new_tokens = 500) # 2d tensor
sample_output = _tokenizer.character_decode(sample[0].tolist(), dict_to_token)
print(sample_output)