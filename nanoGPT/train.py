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

SIZE_CONTEXT=8
SIZE_BATCH=4

def demo_context_recursion(tensor):
    print(f"{'context':.<{SIZE_BATCH*SIZE_CONTEXT}} -> target")
    for i in range(SIZE_CONTEXT):
        context = tensor[:i+1] # stop at i
        target = tensor[i+1]
        print(f"{str(context.tolist()):.<{SIZE_BATCH*SIZE_CONTEXT}} -> {target.item()}")

    return

demo_context_recursion(vector_tokens[:SIZE_CONTEXT+1])

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

# Bigram model
model = models.BigramLanguageModel(len(dict_to_idx)) # size_vocab
xtr, ytr = next(iter(dataloader_train))

logits, loss = model(xtr, ytr)

# train model
optimizer = torch.optim.AdamW(params = model.parameters(), lr = 1e-4)
NUM_EPOCHS=0 # runs through the dataset
for epoch in range(NUM_EPOCHS):
    for batch in dataloader_train:
        xtr, ytr = batch
        model.zero_grad()

        log_probs, loss = model(xtr, ytr)

        loss.backward()
        optimizer.step()
    print(loss)

tokens_single_char = torch.zeros((1,1), dtype = torch.long)
sample = model.generate(input = tokens_single_char, max_new_tokens = 100) # 2d tensor
sample_output = _tokenizer.character_decode(sample[0].tolist(), dict_to_token)
print(sample_output)