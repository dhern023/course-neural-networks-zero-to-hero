"""
Reconstruct the 2003 paper with
    Y = softmax \circ ((tanh \circ (C[X] @ W1 + b1)) @ W2 + b2)
    where 
        X is the tokenized inputs,
        C is the (|V|, k) mapping that embeds X into a lower dimensional space
        W1 is the (k, hidden) weight matrix
        b1 is the (hidden, 1) bias vector
        W2 is the (hidden, |V|) weight matrix
        b2 = is the (|V|) bias vector
    
    i.e,
        Y = softmax \circ linear \circ tanh \circ linear \circ embedding

NOTE: We do things a litte inefficiently in order to demonstrate
"""
import matplotlib.pyplot as plt
import pathlib
import torch
import torch.nn
import torch.nn.functional
import tqdm

DIR_READ = pathlib.Path(__file__).resolve().parent
DIR_OUT = pathlib.Path(__file__).resolve().parents[1] / "out" / "makemore-mlp"
DIR_OUT.mkdir(exist_ok=True,parents=True)

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

dict_token_to_ix, dict_ix_to_token = construct_map_token_to_index("".join(words))

list_tokens_extra = ["."]
for token in list_tokens_extra:
    dict_token_to_ix[token] = len(dict_token_to_ix)
    dict_ix_to_token[len(dict_ix_to_token)] = token

# Construct our dataset via windows ---------------------------------------------------------------
list_documents = [ list(string) + ["."] for string in words ] # different

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
for document in list_documents[:5]:
    for i in range(len(document)): # num_ngrams
        print((["."]*(SIZE_CONTEXT-1) + document)[i:i+SIZE_CONTEXT], list_vectors[i])

matrix_ngrams = torch.tensor(list_vectors, dtype = torch.int64)
xs = matrix_ngrams[:,0:SIZE_CONTEXT-1] #
ys = matrix_ngrams[:,-1] # vector

# Constructing the parameters --------------------------------------------------------------------------------
g = torch.Generator().manual_seed(2147483647) # default seed from the torch docs

SIZE_DIMENSION = 2
SIZE_HIDDEN = 100
C = torch.randn(size=(len(dict_ix_to_token),SIZE_DIMENSION), generator=g)
W1 = torch.randn(size=(SIZE_CONTEXT-1, SIZE_DIMENSION, SIZE_HIDDEN), generator=g)
b1 = torch.randn(SIZE_HIDDEN, generator=g)
W2 = torch.randn(size=(SIZE_HIDDEN, len(dict_ix_to_token)), generator=g)
b2 = torch.randn(len(dict_ix_to_token), generator=g)
parameters = [C, W1, b1, W2, b2]
 # copy data but not gradients
parameters_demo_learning_rate = [t.clone() for t in parameters]
parameters_demo_learning_rate_decay = [t.clone() for t in parameters]
parameters_demo_train_test_split = [t.clone() for t in parameters]

num_epochs = 1000
SIZE_BATCH = 32

# Identify the best learning rates visually ----------------------------------------------------------------
learning_rate_exponents = torch.linspace(-3, 0, num_epochs) # to search exponentially
learning_rates = 10**learning_rate_exponents

dict_learning_rates = {
    "rate": learning_rates,
    "exponent": learning_rate_exponents,
    "loss": torch.zeros(len(learning_rates))
}
for p in parameters:
    p.requires_grad=True
for i in range(len(learning_rates)):
    # mini-batch
    indices = torch.randint(0, xs.shape[0], (SIZE_BATCH,))

    embed = C[xs[indices]] # shape (num_ngrams[indices], SIZE_CONTEXT-1, SIZE_DIMENSION)
    h = torch.tanh(torch.einsum('ijk,jkl -> il', embed, W1) + b1) # hidden states
    logits = h @ W2 + b2
    loss = torch.nn.functional.cross_entropy(logits, ys[indices])
    # backward pass
    for p in parameters:
        p.grad = None
    loss.backward()

    # update parameters
    learning_rate = dict_learning_rates["rate"][i]
    for p in parameters:
        p.data += -learning_rate * p.grad # going against the gradient reduces the loss

    # collect stats
    dict_learning_rates["loss"][i] = loss.item()

plt.plot(dict_learning_rates["rate"], dict_learning_rates["loss"])
plt.savefig(DIR_OUT / "rates_vs_losses.png")
plt.close()

plt.plot(dict_learning_rates["exponent"], dict_learning_rates["loss"])
plt.savefig(DIR_OUT / "exponents_vs_losses.png")
plt.close()

# Finding the best learning rates analytically ----------------------------------------------------

def first_derivative(xs, ys):
    dydx_full = torch.gradient(ys, spacing=(xs,))[0]
    return dydx_full

derivatives_rates = first_derivative(
    dict_learning_rates["rate"],
    dict_learning_rates["loss"]
)

plt.plot(dict_learning_rates["rate"], derivatives_rates)
plt.plot(dict_learning_rates["rate"], dict_learning_rates["loss"])
plt.savefig(DIR_OUT / "rates_vs_losses and vs_loss_derivative.png")
plt.close()
plt.plot(dict_learning_rates["exponent"], derivatives_rates)
plt.plot(dict_learning_rates["exponent"], dict_learning_rates["loss"])
plt.savefig(DIR_OUT / "exponents_vs_losses and vs_loss_derivative.png")
plt.close()

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
    h = torch.tanh(torch.einsum('ijk,jkl -> il', embed, list_parameters[1]) + list_parameters[2]) # hidden states
    logits = h @ list_parameters[3] + list_parameters[4]
    loss = torch.nn.functional.cross_entropy(logits, Y)

    return loss

def find_lower_n_percentile(tensor, n = 0.05):
    """
    Get mask in the bottom n% of all absolute values

    Useful if the tensor is a set of derivatives, since we may want
        x values that produce small derivatives that may not be 0 (ideally),
        but still produce small, steady changes
    """
    percentile = torch.quantile(torch.abs(tensor), n)
    return torch.abs(tensor) < percentile

percent_upper_bound = 0.1
mask = find_lower_n_percentile(derivatives_rates, percent_upper_bound)
dict_learning_rates_lower_percentile = {
    "rate": dict_learning_rates["rate"][mask],
    # "exponent": dict_learning_rates["exponent"][find_lower_n_percentile(derivatives_exponents, percentile)],
    "loss": torch.zeros(mask.sum())
}
for p in parameters_demo_learning_rate:
    p.requires_grad=True
for i in range(len(dict_learning_rates_lower_percentile["rate"])):
    # mini-batch
    indices = torch.randint(0, xs.shape[0], (SIZE_BATCH,))
    loss = forward_pass(xs[indices], ys[indices], parameters_demo_learning_rate)
    # backward pass
    for p in parameters_demo_learning_rate:
        p.grad = None
    loss.backward()

    # update parameters
    learning_rate = dict_learning_rates_lower_percentile["rate"][i]
    for p in parameters_demo_learning_rate:
        p.data += -learning_rate * p.grad # going against the gradient reduces the loss

    # collect stats
    dict_learning_rates_lower_percentile["loss"][i] = loss.item()

plt.plot(dict_learning_rates_lower_percentile["rate"], dict_learning_rates_lower_percentile["loss"])
plt.savefig(DIR_OUT / f"rates_vs_loss {int(percent_upper_bound*100)}th percentile.png") # should be close to a flat line
plt.close()

plt.plot(dict_learning_rates_lower_percentile["rate"], dict_learning_rates_lower_percentile["loss"])
plt.plot(dict_learning_rates_lower_percentile["rate"], derivatives_rates[mask])
plt.savefig(DIR_OUT / f"rates_vs_loss and vs_derivatives {int(percent_upper_bound*100)}th_percentile.png")
plt.close()

# Learning rate decay -----------------------------------------------------------------------------

LEARNING_RATE = 0.1
num_epochs = 2000
epoch_decay = num_epochs*0.9
dict_loss_decay = {
    "loss": []
}
print("TRAINING WITH LEARNING RATE DECAY")
for p in parameters_demo_learning_rate_decay:
    p.requires_grad=True
for i in range(num_epochs):
    # mini-batch
    indices = torch.randint(0, xs.shape[0], (SIZE_BATCH,))
    loss = forward_pass(xs[indices], ys[indices], parameters_demo_learning_rate_decay)
    # backward pass
    for p in parameters_demo_learning_rate_decay:
        p.grad = None
    loss.backward()

    # update parameters
    if i == epoch_decay:
        print("Loss before learning rate decay:", loss.item())
        LEARNING_RATE = LEARNING_RATE * 0.1
    for p in parameters_demo_learning_rate_decay:
        p.data += -LEARNING_RATE * p.grad # going against the gradient reduces the loss
    dict_loss_decay["loss"].append(loss.item())

print("Loss after learning rate decay:", loss.item())
plt.plot(range(len(dict_loss_decay["loss"])), dict_loss_decay["loss"])
plt.savefig(DIR_OUT/f"train-loss-decay.png")
plt.close()

# Train/Test Split --------------------------------------------------------------------------------

import random
random.seed(42)
random.shuffle(xs)
random.shuffle(matrix_ngrams.clone())
xs = matrix_ngrams[:,0:SIZE_CONTEXT-1]
ys = matrix_ngrams[:,-1] # vector

n1 = int(0.8 * xs.shape[0])
n2 = int(0.9 * xs.shape[0])
Xtr, Xdev, Xts = xs.tensor_split((n1, n2), dim=0)
Ytr, Ydev, Yts = ys.tensor_split((n1, n2), dim=0)

def train_neural_net(xtr, ytr, list_parameters, num_epochs = 30000, size_batch = 35):
    list_steps = []
    list_loss = []

    LEARNING_RATE = 0.1 # discovered empirocally
    epoch_decay = num_epochs*0.9 # after 90% of epochs
    print("TRAINING TRAIN SET WITH LEARNING RATE DECAY")
    for p in list_parameters:
        p.requires_grad=True
    for i in range(num_epochs):
        # mini-batch
        indices = torch.randint(0, xtr.shape[0], (size_batch,))
        loss = forward_pass(xtr[indices], ytr[indices], list_parameters)
        # backward pass
        for p in list_parameters:
            p.grad = None
        loss.backward()

        # update parameters
        if i == epoch_decay:
            print("Loss before learning rate decay:", loss.item())
            LEARNING_RATE = LEARNING_RATE * 0.1
        for p in list_parameters:
            p.data += -LEARNING_RATE * p.grad # going against the gradient reduces the loss
        list_steps.append(i)
        list_loss.append(loss.item())

    print("Loss after learning rate decay:", loss.item())
    plt.plot(list_steps, list_loss)
    plt.savefig(DIR_OUT / f"train-loss-{sum(p.numel() for p in list_parameters)}.png")
    plt.close()

    return list_parameters

parameters_demo_train_test_split = train_neural_net(Xtr, Ytr, parameters_demo_train_test_split, num_epochs)
loss_train = forward_pass(Xtr, Ytr, parameters_demo_train_test_split)
loss_dev = forward_pass(Xdev, Ydev, parameters_demo_train_test_split)
print("Training loss:", loss_train.item())
print("Validate loss:", loss_dev.item())

# Exercise ----------------------------------------------------------------------------------------
dict_hyperparameters = {
    "DIMENSIONS": [2],
    "HIDDENS": [100]
}
import itertools

BEST_PARAMETERS = None
BEST_LOSS = float('inf')
list_list_parameters = []
num_epochs = 30000
print("HYPERPARAMETER-TRAINING")
for (SIZE_DIMENSION, SIZE_HIDDEN) in tqdm.tqdm(itertools.product(*list(dict_hyperparameters.values()))):
    C = torch.randn(size=(len(dict_ix_to_token),SIZE_DIMENSION), generator=g)
    W1 = torch.randn(size=(SIZE_CONTEXT-1, SIZE_DIMENSION, SIZE_HIDDEN), generator=g)
    b1 = torch.randn(SIZE_HIDDEN, generator=g)
    W2 = torch.randn(size=(SIZE_HIDDEN, len(dict_ix_to_token)), generator=g)
    b2 = torch.randn(len(dict_ix_to_token), generator=g)
    list_parameters = [C, W1, b1, W2, b2]

    print(sum(p.numel() for p in list_parameters))
    list_parameters_updated = train_neural_net(Xtr, Ytr, list_parameters, num_epochs)
    loss_train = forward_pass(Xtr, Ytr, list_parameters_updated)
    loss_dev = forward_pass(Xdev, Ydev, list_parameters_updated)
    print("Training loss:", loss_train.item())
    print("Validate loss:", loss_dev.item())
    if loss_dev.item() < BEST_LOSS:
        print(f"Best loss updated for (DIMENSION, HIDDEN): {SIZE_DIMENSION, SIZE_HIDDEN}")
        BEST_LOSS = loss_dev.item()
        BEST_PARAMETERS = (SIZE_DIMENSION, SIZE_HIDDEN)

print("Best loss",BEST_LOSS, "with (DIMENSION, HIDDEN)", BEST_PARAMETERS)

# Sampling from the model
def sample_from_model(list_parameters, num_samples = 20):
    print("Sampling from the model")
    g = torch.Generator().manual_seed(2147483647 + 10) # default seed from the torch docs
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

samples = sample_from_model(list_parameters, num_samples=20)
for sample in samples:
    print(sample)