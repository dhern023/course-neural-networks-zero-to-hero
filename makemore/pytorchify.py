"""
Different from the actual implementation in that we use pytorch completely.
I'm interested in the activation statistics
TODO: Use the safetensor for reading the dataset
"""
import matplotlib.pyplot as plt
import matplotlib.lines
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

    def __init__(self, num_embeddings, size_dimension, size_input, size_hidden, size_output, bool_initialize=False):
        """
        """
        super(MLP, self).__init__() # override class
        self.C = torch.nn.Embedding(num_embeddings,embedding_dim=size_dimension)
        self.W = torch.nn.Linear(size_input*size_dimension, size_hidden)
        self.hiddens = torch.nn.ModuleList(
            [torch.nn.Linear(size_hidden, size_hidden) for _ in range(4)]
        )
        self.logits = torch.nn.Linear(size_hidden, size_output)
        self.activation = torch.nn.Tanh()

        if bool_initialize:
            self.init_weights() # avoid the vanishing gradient problem

    def init_weights(self):
        """
        Pytorch torch.nn.Linear layers are automatically initialized using a variant of Kaiming (He) initialization.
        Problem is this works poorly with tanh, whose activations benefit more from Xavier (Godot).
        As is, we would run into the vanishing gradient problem if we didn't invoke batch_normalization
        """
        torch.nn.init.kaiming_uniform_(self.W.weight, nonlinearity='tanh')
        torch.nn.init.zeros_(self.W.bias)
        for hidden in self.hiddens:
            torch.nn.init.kaiming_uniform_(hidden.weight, nonlinearity='tanh')
            torch.nn.init.zeros_(hidden.bias)
        torch.nn.init.kaiming_uniform_(self.logits.weight, nonlinearity='tanh')
        torch.nn.init.zeros_(self.logits.bias)

    def forward(self, vector_input):
        """
        Requires vector of (num_ngrams[indices], size_input)
        """
        embed = self.C(vector_input) # (num_ngrams[indices], size_input, size_dimension)
        embed = embed.view(embed.shape[0], -1) # reshape
        layer = self.W(embed) # (size_dimension, size_hidden)
        layer = self.activation(layer)        

        for hidden in self.hiddens:
            layer = hidden(layer) # (size_hidden, size_hidden)
            layer = self.activation(layer)

        layer = self.logits(layer) # (size_hidden, size_output)

        return layer

def plot_grad_flow(instance_model):
    """
    Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.
    """
    ave_grads = []
    max_grads = []
    layers = []

    for name, parameter in instance_model.named_parameters():
        if parameter.requires_grad and "bias" not in name:
            layers.append(name)
            ave_grads.append(parameter.grad.abs().mean().item())
            max_grads.append(parameter.grad.abs().max().item())

    # Convert lists to tensors
    ave_grads_tensor = torch.tensor(ave_grads)
    max_grads_tensor = torch.tensor(max_grads)

    # Create histogram bins
    bins = torch.linspace(0, len(max_grads), len(max_grads) + 1)

    # Compute histograms
    hist_ave_grads = torch.histogram(ave_grads_tensor, bins=bins)
    hist_max_grads = torch.histogram(max_grads_tensor, bins=bins)

    # Plotting
    plt.bar(hist_max_grads.bin_edges[:-1], hist_max_grads.hist, width=hist_max_grads.bin_edges[1]-hist_max_grads.bin_edges[0], alpha=0.1, lw=1, color="c")
    plt.bar(hist_ave_grads.bin_edges[:-1], hist_ave_grads.hist, width=hist_ave_grads.bin_edges[1]-hist_ave_grads.bin_edges[0], alpha=0.1, lw=1, color="b")
    plt.hlines(0, 0, len(ave_grads), lw=2, color="k")
    plt.xticks(torch.arange(len(ave_grads)), layers, rotation="vertical")
    plt.xlim(left=0, right=len(ave_grads))
    plt.ylim(bottom=-0.001, top=0.02)  # zoom in on the lower gradient regions
    plt.xlabel("Layers")
    plt.ylabel("Average Gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.legend([matplotlib.lines.Line2D([0], [0], color="c", lw=4),
                matplotlib.lines.Line2D([0], [0], color="b", lw=4),
                matplotlib.lines.Line2D([0], [0], color="k", lw=4)], ['max-gradient', 'mean-gradient', 'zero-gradient'])
    plt.savefig(DIR_OUT / "activation-flow.png")


def train_model(instance_model, num_epochs, dataloader):
    """
    Training loop has been modified such that the epochs are the number of times
        we pass over the entire sampled dataset based on the dataloader    

    https://web.stanford.edu/~nanbhas/blog/forward-hooks-pytorch/

    Use hooks to save the activation's values and gradients each time it's called
        for the linear layer and each hidden layer
    """
    LEARNING_RATE = 0.1 # discovered empirocally
    optimizer = torch.optim.SGD(instance_model.parameters(), lr=LEARNING_RATE)
    loss_function = torch.nn.CrossEntropyLoss()

    list_loss = []
    # tensor_losses = torch.zeros(size=(num_epochs, len(Xtr)))
    epoch_decay = num_epochs*0.9
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[epoch_decay], gamma=0.1)

    # Hooks to store the activations
    dict_activations = {}
    def get_activation(name):
        # the hook signature
        def hook(model, input, output):
            dict_activations.setdefault(name, [])
            dict_activations[name].append(output.detach())
        return hook

    dict_gradients = {}
    def get_gradient(name):
        # the hook signature
        def hook(module, grad_input, grad_output):
            dict_gradients.setdefault(name, [])
            dict_gradients[name].append(grad_output[0])
        return hook

    # register forward hooks on the layers of choice
    h1 = model.activation.register_forward_hook(get_activation('activation'))
    b1 = model.activation.register_full_backward_hook(get_gradient('activation'))
    dict_out = {
        "activations": [],
        "gradients": []
    }    

    for i in tqdm.tqdm(range(num_epochs), total = num_epochs//1000):
        for batch in dataloader:
            model.zero_grad() # zero out the gradients
            xtr, ytr = batch
            log_probs = model(xtr)
            loss = loss_function(log_probs, ytr)

            loss.backward() # compute gradient
            optimizer.step() # update parameters

            list_loss.append(loss.log10().item())
            dict_out["activations"].append(dict_activations['activation'])
            dict_out["gradients"].append(dict_gradients['activation'])

            # detach the hooks to only save after the first run
            h1.remove()
            b1.remove()

        if i == epoch_decay:
            print("\tLoss before learning rate decay:", loss.item())
        scheduler.step()

    print("\tLoss after learning rate decay:", loss.item())

    return instance_model, list_loss, dict_out

SIZE_CONTEXT = 4
SIZE_DIMENSION=10
SIZE_HIDDEN=200
dataset = torch.utils.data.TensorDataset(Xtr, Ytr)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
NUM_EPOCHS = int(20) # int(1)  # int(2e4) # one per each dataset run

bool_initialize_weights = True
model = MLP(
    num_embeddings=len(dict_ix_to_token), 
    size_dimension=SIZE_DIMENSION, 
    size_input=SIZE_CONTEXT-1,
    size_hidden=SIZE_HIDDEN,
    size_output=len(dict_ix_to_token),
    bool_initialize=bool_initialize_weights
)
model, list_loss, dict_snapshots = train_model(model, NUM_EPOCHS, dataloader)
plt.plot(list_loss)
plt.savefig(DIR_OUT / "train-loss-decay-pytorch")
plt.close()

plot_grad_flow(model)

# Forward & Backward pass activation statistics ===================================================

def plot_activation_layer_statistics(list_tensors, saturated_bound = 0.9, tag="Activation"):
    """
    Assumes you stored these tensors using a hook in pytorch
    """
    out = plt.figure()
    plt.title(f'{tag} Histogram for Each Layer')
    plt.xlabel(f'{tag} Value')
    plt.ylabel('Frequency')
    for i, activation in enumerate(list_tensors):
        t = activation
        saturated_percent = (t.abs() > saturated_bound).float().mean().mul_(100).item()
        print(f"{tag} layer {i}: mean {t.mean():0.2f}, std {t.std():0.2f}, saturated: {saturated_percent}%")
        tensors_hist = torch.histogram(t.detach().cpu(), bins=50, density=True) 
        plt.plot(tensors_hist.bin_edges[:-1], tensors_hist.hist, label=f'For Layer {i+1}')
    plt.legend()

    return out

instance_figure = plot_activation_layer_statistics(
    dict_snapshots["activations"][0],
    saturated_bound=0.9,
    tag="Activation"
)
if bool_initialize_weights:
    instance_figure.savefig(DIR_OUT / f"activation-distibution-properly-initialized")
else:
    instance_figure.savefig(DIR_OUT / f"activation-distibution-vanishing-gradients")

instance_figure = plot_activation_layer_statistics(
    dict_snapshots["gradients"][0],
    saturated_bound=0.9,
    tag="Activation Gradient"
)
if bool_initialize_weights:
    instance_figure.savefig(DIR_OUT / "activation-gradient-distibution-properly-initialized")
else:
    instance_figure.savefig(DIR_OUT / "activation-gradient-distibution-vanishing-gradients")