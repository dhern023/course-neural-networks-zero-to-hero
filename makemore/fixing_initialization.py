"""
Covers the Activations & Gradients section

Most of the fixes amount to setting up the paremeters properly,
    so we setup a function to take in a list of parameters with hard-coded hyperparameters
"""
import matplotlib.pyplot as plt
import pathlib
import random
import torch
import torch.nn
import torch.nn.functional
import tqdm

# Defaults
DIRNAME_OUT = "makemore-fixing-initialization"
DIR_READ = pathlib.Path(__file__).resolve().parent
DIR_OUT = pathlib.Path(__file__).resolve().parents[1] / "out" / DIRNAME_OUT
DIR_OUT.mkdir(exist_ok=True, parents=True)

torch.set_default_device("cuda" if torch.cuda.is_available() else "cpu")
g = torch.Generator(device=torch.get_default_device()).manual_seed(2147483647) # default seed from the torch docs, shares memory based on docs

# reading data to tokens
def load_txt_to_list(fname):
    path = pathlib.Path(fname)
    if not path.exists():
        raise Exception(f"{fname} not found!")

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

# Same as in the video - before hypertuning to just demonstrate -----------------------------------
SIZE_DIMENSION=10
SIZE_HIDDEN=200

C = torch.randn(size=(len(dict_ix_to_token),SIZE_DIMENSION), generator=g)
W1 = torch.randn(size=(SIZE_CONTEXT-1, SIZE_DIMENSION, SIZE_HIDDEN), generator=g)
b1 = torch.randn(SIZE_HIDDEN, generator=g)
W2 = torch.randn(size=(SIZE_HIDDEN, len(dict_ix_to_token)), generator=g)
b2 = torch.randn(len(dict_ix_to_token), generator=g)
list_parameters_starter = [C, W1, b1, W2, b2]

def forward_pass(X, Y, list_parameters):
    """
    logits = tanh \circ (C[X] @ W1 + b1)) @ W2 + b2
    loss = cross_entropy(logits, Y)
    where
        X is the tokenized inputs, Y is the tokenized outputs
        C is the (|V|, k) mapping that embeds X into a lower dimensional space
        W1 is the (k, hidden) weight matrix
        b1 is the (hidden, 1) bias vector
        W2 is the (hidden, |V|) weight matrix
        b2 = is the (|V|) bias vector

    i.e,
        logits = linear \circ tanh \circ linear \circ embedding
    """
    embed = list_parameters[0][X] # shape (num_ngrams[indices], SIZE_CONTEXT-1, SIZE_DIMENSION)
    hidden = torch.einsum('ijk,jkl -> il', embed, list_parameters[1]) + list_parameters[2] # hidden states
    h = torch.tanh(hidden) # activated hidden states
    logits = h @ list_parameters[3] + list_parameters[4]
    loss = torch.nn.functional.cross_entropy(logits, Y)

    return loss

def train_model(list_parameters, num_epochs=int(5e3)):
    SIZE_BATCH=32

    list_steps = []
    list_loss = []

    LEARNING_RATE = 0.1 # discovered empirocally
    epoch_decay = num_epochs*0.9 # after 90% of epochs
    for p in list_parameters:
        p.requires_grad=True
    for i in tqdm.tqdm(range(num_epochs), total = num_epochs//1000):
        # mini-batch
        indices = torch.randint(0, Xtr.shape[0], (SIZE_BATCH,))
        loss = forward_pass(Xtr[indices], Ytr[indices], list_parameters)
        # backward pass
        for p in list_parameters:
            p.grad = None
        loss.backward()

        # learning rate decay
        if i == epoch_decay:
            print("\tLoss before learning rate decay:", loss.item())
            LEARNING_RATE = LEARNING_RATE * 0.1

        # update parameters
        for p in list_parameters:
            p.data += -LEARNING_RATE * p.grad # going against the gradient reduces the loss
        list_loss.append(loss.log10().item())

    print("\tLoss after learning rate decay:", loss.item())

    with torch.no_grad():
        loss_train = forward_pass(Xtr, Ytr, list_parameters)
        loss_dev = forward_pass(Xdev, Ydev, list_parameters)
        print("\tTraining loss:", loss_train.item())
        print("\tValidate loss:", loss_dev.item())

    return list_parameters, list_loss

def sample_from_model(list_parameters, num_samples = 20):
    print("Sampling from the model")
    g = torch.Generator(device=torch.get_default_device()).manual_seed(2147483647 + 10) # default seed from the torch docs
    list_out = []
    for i in range(num_samples):
        temp_out = []
        context = [dict_token_to_ix['.']] * (SIZE_CONTEXT-1)
        while True:
            embed = list_parameters[0][torch.tensor(context).view(1,-1)] # shape (1, SIZE_CONTEXT-1, SIZE_DIMENSION)
            h = torch.tanh(torch.einsum('ijk,jkl -> il', embed, list_parameters[1]) + list_parameters[2]) # hidden states
            logits = h @ list_parameters[3] + list_parameters[4]
            probs = torch.nn.functional.softmax(logits, dim=1)
            ix = torch.multinomial(probs, num_samples=1, replacement=True, generator=g).item()
            context = context[1:] + [ix]
            temp_out.append(dict_ix_to_token[ix])
            if ix == dict_token_to_ix['.']:
                break
        out = ''.join(temp_out)
        list_out.append(out)

    return list_out

bool_demo_train_starter = True
if bool_demo_train_starter:
    print("TRAINING NETWORK - From Last Time")
    NUM_EPOCHS = int(5e3) # int(2e4) # int(2e5)
    list_parameters_trained, list_loss = train_model(list_parameters_starter, NUM_EPOCHS)
    samples = sample_from_model(list_parameters_trained, num_samples=20)
    print(samples, sep = "\n")
    plt.plot(list_loss)
    plt.savefig(DIR_OUT / "train-loss-decay.png")
    plt.close()

# Fixing the initial loss -------------------------------------------------------------------------
def demo_initial_loss():
    """
    Since our neural network is a classifier, we expect initial loss
        to be roughly NLL(1/|V|) ~ 2.29 for |V| = 26 tokens + 1 stop token
    """
    num_labels = 4
    list_logits = [
        torch.zeros(num_labels),
        torch.rand(num_labels, generator=g), # same loss as above
        torch.tensor([0.0, 0.0, 5.0, 0.0]), # high probability of label, low loss
        torch.tensor([0.0, 5.0, 0.0, 0.0]), # low probability of label, high loss
        torch.randn(num_labels, generator=g), # near zero, not great
        torch.randn(num_labels, generator=g) * 10, # worse, higher initial loss
        torch.randn(num_labels, generator=g) * 100, # worse, very high initial loss
    ]
    Y = torch.tensor(2)
    for logits in list_logits:
        probs = torch.softmax(logits, dim=0)
        loss = torch.nn.functional.cross_entropy(logits, Y)
        print("logits:", logits)
        print("probs:", probs)
        print("loss:",loss.item())

demo_initial_loss()

SIZE_DIMENSION=10
SIZE_HIDDEN=200

C = torch.randn(size=(len(dict_ix_to_token),SIZE_DIMENSION), generator=g)
W1 = torch.randn(size=(SIZE_CONTEXT-1, SIZE_DIMENSION, SIZE_HIDDEN), generator=g)
b1 = torch.randn(SIZE_HIDDEN, generator=g)
W2 = torch.randn(size=(SIZE_HIDDEN, len(dict_ix_to_token)), generator=g) * 0.01
b2 = torch.randn(len(dict_ix_to_token), generator=g) * 0
list_parameters_fix_initial = [C, W1, b1, W2, b2]

bool_demo_train_fix_initial = True
if bool_demo_train_fix_initial:
    print("TRAINING NETWORK - Corrected Initialization")
    NUM_EPOCHS = int(1e4) # int(2e4) # int(2e5)
    list_parameters, list_loss = train_model(list_parameters_fix_initial, NUM_EPOCHS)
    samples = sample_from_model(list_parameters, num_samples=20)
    print(samples, sep = "\n")
    plt.plot(list_loss)
    plt.savefig(DIR_OUT / "train-loss-decay-fixed-initial.png")
    plt.close()

# Fixing saturated hidden layer -------------------------------------------------------------------
def demo_estimate_gain():
    """
    Calculate 1 / E[f(Z)^2] where Z is the standard normal, f is an activation function
    """
    return 1 / numpy.mean(numpy.tanh(numpy.random.standard_normal(int(1e7)))**2)

def demo_hidden_layer_saturation(xtr):
    """
    NOTE: We're constructing these activations using the entire dataset, not a batch
    """
    C = torch.randn(size=(len(dict_ix_to_token),SIZE_DIMENSION), generator=g)
    W1 = torch.randn(size=(SIZE_CONTEXT-1, SIZE_DIMENSION, SIZE_HIDDEN), generator=g)
    b1 = torch.randn(SIZE_HIDDEN, generator=g)
    W2 = torch.randn(size=(SIZE_HIDDEN, len(dict_ix_to_token)), generator=g) * 0.01
    b2 = torch.randn(len(dict_ix_to_token), generator=g) * 0
    list_parameters = [C, W1, b1, W2, b2]

    embed = list_parameters[0][xtr] # shape (num_ngrams[indices], SIZE_CONTEXT-1, SIZE_DIMENSION)
    hidden = torch.einsum('ijk,jkl -> il', embed, list_parameters[1]) + list_parameters[2] # hidden states
    h = torch.tanh(hidden) # activated hidden states

    # grad of tanh is 1-t**2, and a very active hidden layer
    # means we are mostly passing t in {-1,1}, effectively killing the gradient
    # also, a grad of t = 0 implies the grad is inactive
    tensors_hist = torch.histogram(h.cpu(), bins=50)
    plt.bar(tensors_hist.bin_edges[:-1], tensors_hist.hist, width=tensors_hist.bin_edges[1]-tensors_hist.bin_edges[0])
    plt.savefig(DIR_OUT / "hidden-layer-active.png")
    plt.close()

    # why? most of these preactivations are going to be squashed to {-1,1}
    tensors_hist = torch.histogram(hidden.cpu(), bins=50)
    plt.bar(tensors_hist.bin_edges[:-1], tensors_hist.hist, width=tensors_hist.bin_edges[1]-tensors_hist.bin_edges[0])
    plt.savefig(DIR_OUT / "preactivations-outside-tanh-range.png")
    plt.close()

    plt.imshow((h.abs() > 0.99)[:32].cpu(), cmap="gray", interpolation="nearest")
    plt.tight_layout()
    plt.savefig(DIR_OUT / "gradient-map-destroyed.png", dpi=300, bbox_inches='tight')
    plt.close()

    # a column of h = {1|-1} would be a "dead" neuron
    plt.title("Dead Neuron Example - Small Batch Size & Hidden Layer")
    plt.imshow((h.abs() > 0.99)[:4, :30].cpu(), cmap="gray", interpolation="nearest")
    plt.tight_layout()
    plt.savefig(DIR_OUT / "dead-neuron-gradient-map-batch-size-4-hidden-size-30.png", bbox_inches='tight')
    plt.close()

def demo_hidden_layer_desaturation(xtr):
    """
    Squashing the initial weights reduces the amount of saturated values that go to {-1,1} due to rounding
    NOTE: We're constructing these graphs using the entire dataset, not a batch
    """
    C = torch.randn(size=(len(dict_ix_to_token),SIZE_DIMENSION), generator=g)
    W1 = torch.randn(size=(SIZE_CONTEXT-1, SIZE_DIMENSION, SIZE_HIDDEN), generator=g) * 0.15
    b1 = torch.randn(SIZE_HIDDEN, generator=g) * 0.01
    W2 = torch.randn(size=(SIZE_HIDDEN, len(dict_ix_to_token)), generator=g) * 0.01
    b2 = torch.randn(len(dict_ix_to_token), generator=g) * 0
    list_parameters = [C, W1, b1, W2, b2]

    embed = list_parameters[0][xtr] # shape (num_ngrams[indices], SIZE_CONTEXT-1, SIZE_DIMENSION)
    hidden = torch.einsum('ijk,jkl -> il', embed, list_parameters[1]) + list_parameters[2] # hidden states
    h = torch.tanh(hidden) # activated hidden states

    # grad of tanh is 1-tanh(x)**2, and a very active hidden layer
    # means we are mostly passing tanh(x) in {-1,1}, effectively killing the gradient
    # also, a grad of t = 0 implies the grad is inactive
    tensors_hist = torch.histogram(h.cpu(), bins=50)
    plt.bar(tensors_hist.bin_edges[:-1], tensors_hist.hist, width=tensors_hist.bin_edges[1]-tensors_hist.bin_edges[0])
    plt.savefig(DIR_OUT / "hidden-layer-inactive.png")
    plt.close()

    # why? most of these preactivations are going to be squashed to {-1,1}
    tensors_hist = torch.histogram(hidden.cpu(), bins=50)
    plt.bar(tensors_hist.bin_edges[:-1], tensors_hist.hist, width=tensors_hist.bin_edges[1]-tensors_hist.bin_edges[0])
    plt.savefig(DIR_OUT / "preactivations-inside-tanh-range.png")
    plt.close()

    plt.imshow((h.abs() > 0.99)[:32].cpu(), cmap="gray", interpolation="nearest")
    plt.tight_layout()
    plt.savefig(DIR_OUT / "gradient-map-good.png", dpi=300, bbox_inches='tight')
    plt.close()

demo_hidden_layer_saturation(Xtr)
demo_hidden_layer_desaturation(Xtr)

# Kaiming Unit
def demo_gaussian_spread():
    """
    Note: It's possible the sample sizes are not large enough to really get E[X]E[W] = E[Y]
    """
    gain = 1 # linearity
    num_inputs = 10
    X = torch.randn(size=(1000, num_inputs), generator=g)
    W = torch.randn(size=(num_inputs, 200), generator=g) / gain * (num_inputs ** 0.5)
    Y = X @ W # (1000, 200)
    print(X.mean(), X.std())
    print(W.mean(), W.std())
    print(Y.mean(), Y.std())

    plt.figure(layout='tight')
    # input histogram
    ax = plt.subplot(121)
    tensors_hist = torch.histogram(X.cpu(), bins=50)
    ax.set_title("Std for Gaussian Inputs X")
    ax.bar(tensors_hist.bin_edges[:-1], tensors_hist.hist)
    # output histogram
    ax = plt.subplot(122)
    tensors_hist = torch.histogram(Y.cpu(), bins=50)
    ax.set_title("Std for Gaussian Outputs Y=X@W")
    ax.bar(tensors_hist.bin_edges[:-1], tensors_hist.hist)
    plt.savefig(DIR_OUT / "std-visualization.png")
    plt.close()

demo_gaussian_spread()

gain = 5/3 # tanh activation (TODO: Still haven't been able to verify this)
C = torch.randn(size=(len(dict_ix_to_token),SIZE_DIMENSION), generator=g)
W1 = torch.randn(size=(SIZE_CONTEXT-1, SIZE_DIMENSION, SIZE_HIDDEN), generator=g) * gain / (((SIZE_CONTEXT-1)*SIZE_DIMENSION)**0.5)
b1 = torch.randn(SIZE_HIDDEN, generator=g) * 0.01
W2 = torch.randn(size=(SIZE_HIDDEN, len(dict_ix_to_token)), generator=g) * 0.01
b2 = torch.randn(len(dict_ix_to_token), generator=g) * 0
list_parameters_fix_saturation = [C, W1, b1, W2, b2]

bool_demo_train_fix_saturation = True
if bool_demo_train_fix_saturation:
    print("TRAINING NETWORK - Corrected Saturation")
    NUM_EPOCHS = int(1e4) # int(2e4) # int(2e5)
    list_parameters, list_loss = train_model(list_parameters_fix_saturation, NUM_EPOCHS)
    samples = sample_from_model(list_parameters, num_samples=20)
    print(samples, sep = "\n")
    plt.plot(list_loss)
    plt.savefig(DIR_OUT / "train-loss-decay-fixed-saturation.png")
    plt.close()
