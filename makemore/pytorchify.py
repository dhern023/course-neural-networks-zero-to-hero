"""
Different from the actual implementation in that we use pytorch completely.
I'm interested in the activation statistics
NOTE: torch.nn.Linear doesn't allow us to easily adjust fan-in, so our 
saturation won't ever be "bad"
"""
import matplotlib.pyplot as plt
import pathlib
import random
import torch
import torch.nn
import torch.nn.functional
import torch.utils.data
import tqdm

# Defaults
DIRNAME_OUT = "makemore-pytorchify"
DIR_READ = pathlib.Path(__file__).resolve().parent
DIR_OUT = pathlib.Path(__file__).resolve().parents[1] / "out" / DIRNAME_OUT
DIR_OUT.mkdir(exist_ok=True, parents=True)

# torch.set_default_device("cuda" if torch.cuda.is_available() else "cpu")
g = torch.Generator().manual_seed(2147483647) # default seed from the torch docs, shares memory based on docs

# reading data to tokens
def load_txt_to_list(fname):
    path = pathlib.Path(fname)
    if not path.exists():
        raise f"{fname} not found!"

    list_out = []
    with open(path, "r") as file:
        list_out = file.read().splitlines()

    return list_out

def construct_map_token_to_index(list_tokens):
    """
    Returns { token : index }

    Map each token in the vocab to a unique integer,
    which will be its index into the Bag of words vector

    TODO: https://en.wikipedia.org/wiki/Feature_hashing
    NOTE: Fatal flaw is set sorting is random, making debugging a little harder
    """
    dict_to_ix = {}
    dict_to_word = {}
    for i, token in enumerate(set(list_tokens)):
        dict_to_ix[token] = i
        dict_to_word[i] = token

    return dict_to_ix, dict_to_word

fname = DIR_READ / "names.txt"
words = load_txt_to_list(fname)
print(words[:8])

dict_token_to_ix, dict_ix_to_token = construct_map_token_to_index("".join(words))

list_tokens_extra = ["."]
for token in list_tokens_extra:
    dict_token_to_ix[token] = len(dict_token_to_ix)
    dict_ix_to_token[len(dict_ix_to_token)] = token

# Calculating the ngrams
list_documents = [ list(string) + ["."] for string in words ]

def construct_vector_ngram(tokens_context, dict_index):
    return list(map(lambda w: dict_index[w], tokens_context))

def construct_n_grams(list_documents, dict_index, size_context, size_prediction=1):
    """
    Initialize an empty window
    Update as you go along

    NOTE: Creates size_context-1 more vectors to account for the stop tokens
    e.g., size_context = 3-1 creates
        [<S>, <S>, <T>]

    TODO: Account for when size_prediction is larger than length(document)
    """
    list_out = []
    vector = [dict_index["."]]*size_context
    for tokens in list_documents:
        # TODO: Right pad tokens to size_prediction
        for i in range(0, len(tokens)):
            vector = vector[size_prediction:] + construct_vector_ngram(tokens[i:i+size_prediction], dict_index)
            list_out.append(vector)
    return list_out

SIZE_CONTEXT = 4 # SIZE_NGRAMS+1 or BLOCKSIZE
list_vectors = construct_n_grams(list_documents, dict_token_to_ix, size_context=SIZE_CONTEXT)
# Verify the mapping
for document in list_documents[:3]:
    for i in range(len(document)): # num_ngrams
        print((["."]*(SIZE_CONTEXT-1) + document)[i:i+SIZE_CONTEXT], list_vectors[i])

matrix_ngrams = torch.tensor(list_vectors, dtype = torch.int64)

# Train/Test Split
random.seed(42)
random.shuffle(matrix_ngrams.clone())
xs = matrix_ngrams[:,0:SIZE_CONTEXT-1]
ys = matrix_ngrams[:,-1] # vector

n1 = int(0.8 * xs.shape[0])
n2 = int(0.9 * xs.shape[0])
Xtr, Xdev, Xts = xs.tensor_split((n1, n2), dim=0)
Ytr, Ydev, Yts = ys.tensor_split((n1, n2), dim=0)

# Recreate using Pytorch API ----------------------------------------------------------------------

class MLP(torch.nn.Module):

    def __init__(self, num_embeddings, size_dimension, size_input, size_hidden, size_output):
        super(MLP, self).__init__() # override class
        self.C = torch.nn.Embedding(num_embeddings,embedding_dim=size_dimension)
        self.W = torch.nn.Linear(size_input*size_dimension, size_hidden)
        self.hiddens = torch.nn.ModuleList(
            [torch.nn.Linear(size_hidden, size_hidden) for _ in range(4)]
        )
        self.logits = torch.nn.Linear(size_hidden, size_output)
        self.activation = torch.nn.Tanh()

    def forward(self, vector_input):
        """
        Requires vector of (num_ngrams[indices], size_input)
        """
        activations = []

        embed = self.C(vector_input) # (num_ngrams[indices], size_input, size_dimension)
        embed = embed.view(embed.shape[0], -1) # reshape
        layer = self.W(embed) # (size_dimension, size_hidden)
        layer = self.activation(layer)
        activations.append(layer)

        for hidden in self.hiddens:
            layer = hidden(layer) # (size_hidden, size_hidden)
            layer = self.activation(layer)
            activations.append(layer)

        layer = self.logits(layer) # (size_hidden, size_output)

        return layer, activations

def train_model(instance_model, num_epochs, dataloader):

    LEARNING_RATE = 0.1 # discovered empirocally
    optimizer = torch.optim.SGD(instance_model.parameters(), lr=LEARNING_RATE)
    loss_function = torch.nn.CrossEntropyLoss()

    list_loss = []
    # tensor_losses = torch.zeros(size=(num_epochs, len(Xtr)))
    epoch_decay = num_epochs*0.9
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[epoch_decay], gamma=0.1)
    list_activations_all = []
    for i in tqdm.tqdm(range(num_epochs), total = num_epochs//1000):
        for batch in dataloader:
            model.zero_grad() # zero out the gradients
            xtr, ytr = batch
            log_probs, activations = model(xtr)
            loss = loss_function(log_probs, ytr)

            loss.backward() # compute gradient
            optimizer.step() # update parameters

            list_loss.append(loss.log10().item())
            list_activations_all.append(activations)
            break

        if i == epoch_decay:
            print("\tLoss before learning rate decay:", loss.item())
        scheduler.step()

    print("\tLoss after learning rate decay:", loss.item())
    return instance_model, list_loss, list_activations_all

SIZE_CONTEXT = 4
SIZE_DIMENSION=10
SIZE_HIDDEN=200
model = MLP(
    num_embeddings=len(dict_ix_to_token), 
    size_dimension=SIZE_DIMENSION, 
    size_input=SIZE_CONTEXT-1,
    size_hidden=SIZE_HIDDEN,
    size_output=len(dict_ix_to_token)
)

dataset = torch.utils.data.TensorDataset(Xtr, Ytr)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
NUM_EPOCHS = int(1) # int(2e4) # int(2e5)
model, list_loss, list_activations_all = train_model(model, NUM_EPOCHS, dataloader)
plt.plot(list_loss)
plt.savefig(DIR_OUT / "train-loss-decay-pytorch.png")
plt.close()

plt.figure()
plt.title(f'Activation Histogram for Each Layers')
plt.xlabel('Activation Value')
plt.ylabel('Frequency')
for i, activation in enumerate(list_activations_all[0]):
    t = activation
    saturated_percent = (t.abs() > 0.90).float().mean().mul_(100).item()
    print(f"activation layer {i}: mean {t.mean():0.2f}, std {t.std():0.2f}, saturated: {t}%")
    tensors_hist = torch.histogram(t.detach().cpu(), bins=50, density=True) 
    plt.plot(tensors_hist.bin_edges[:-1], tensors_hist.hist, label=f'For Layer {i+1}')
plt.legend()
plt.savefig(DIR_OUT / "activation-distibution")

plt.figure()
plt.title(f'Activation Gradient Histogram for Each Layers')
plt.xlabel('Activation Gradient Value')
plt.ylabel('Frequency')
for i, activation in enumerate(list_activations_all[0]):
    t = activation.grad
    saturated_percent = (t.abs() > 0.90).float().mean().mul_(100).item()
    print(f"activation layer {i}: mean {t.mean():0.2f}, std {t.std():0.2f}, saturated: {t}%")
    tensors_hist = torch.histogram(t.detach().cpu(), bins=50, density=True) 
    plt.plot(tensors_hist.bin_edges[:-1], tensors_hist.hist, label=f'For Layer {i+1}')
plt.legend()
plt.savefig(DIR_OUT / "activation-gradient-distibution")