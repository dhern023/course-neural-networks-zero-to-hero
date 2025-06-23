"""
We only implement the running calibration portion of batch_normalization
since we can calculate the full bngain's mean and std.

NOTE: We keep both running and calculated totals of bngain and mean values to highlight they
    are roughly equivalent
"""

import matplotlib.pyplot as plt
import pathlib
import random
import torch
import torch.nn
import torch.nn.functional
import tqdm

# Defaults
DIRNAME_OUT = "makemore-batch-normalization"
DIR_READ = pathlib.Path(__file__).resolve().parent
DIR_OUT = pathlib.Path(__file__).resolve().parents[1] / "out" / DIRNAME_OUT
DIR_OUT.mkdir(exist_ok=True, parents=True)

# torch.set_default_device("cuda" if torch.cuda.is_available() else "cpu")
g = torch.Generator().manual_seed(2147483647) # default seed from the torch docs, shares memory based on docs

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
# b1 = torch.randn(SIZE_HIDDEN, generator=g)
# New Parameters for batch normalization
bngain = torch.ones((1, SIZE_HIDDEN))
bnbias = torch.zeros((1, SIZE_HIDDEN))

W2 = torch.randn(size=(SIZE_HIDDEN, len(dict_ix_to_token)), generator=g)
b2 = torch.randn(len(dict_ix_to_token), generator=g)

bngain_mean_running = torch.zeros((1,SIZE_HIDDEN)) # mean = 0
bngain_std_running = torch.ones((1,SIZE_HIDDEN)) # std = 1

list_parameters_batch_norm = [C, W1, bngain, bnbias, W2, b2]

def forward_pass_batch_norm(X, Y, list_parameters):
    """
    jittered logits = tanh \circ batch_norm \circ (C[X] @ W1)) @ W2 + b2
    loss = cross_entropy(logits, Y)
    where
        X is the tokenized inputs, Y is the tokenized outputs
        C is the (|V|, k) mapping that embeds X into a lower dimensional space
        W1 is the (k, hidden) linear weight matrix
        # b1 is the (hidden, 1) linear bias vector

        gain is the (1, hidden) batch normalization scaling vector
        b is the (1, hidden) batch normalization bias vector 
        
        W2 is the (hidden, |V|) linear weight matrix
        b2 = is the (|V|) linear bias vector

    hidden = (C[X] @ W1 + b1)
    batch_norm = gain * (hidden - hidden.mean) / hidden.std + bias
    E[batch_norm] = gain E[(hidden - hidden.mean) / hidden.std] + bias
    Then batch_norm - E[batch_norm] =\
        gain E[(hidden - hidden.mean) / hidden.std] - gain E[(hidden - hidden.mean) / hidden.std
    So the biases cancel out
    i.e,
        jittered logits = linear \circ tanh \circ batch_norm \circ linear \circ embedding
    """
    embed = list_parameters[0][X] # shape (num_ngrams[indices], SIZE_CONTEXT-1, SIZE_DIMENSION)
    hidden = torch.einsum('ijk,jkl -> il', embed, list_parameters[1]) # + list_parameters[2] # hidden states
    bngain_mean_i = hidden.mean(0,keepdims=True)
    bngain_std_i = hidden.std(0,keepdims=True)
    hidden_batch = list_parameters[2] * (hidden - bngain_mean_i) / bngain_std_i + list_parameters[3]

    with torch.no_grad():
        bngain_mean_running.mul_(0.999).add_(0.001 * bngain_mean_i)
        bngain_std_running.mul_(0.999).add_(0.001 * bngain_std_i)

    h = torch.tanh(hidden_batch) # activated hidden states

    logits = h @ list_parameters[4] + list_parameters[5]
    loss = torch.nn.functional.cross_entropy(logits, Y)

    return loss

def train_model(list_parameters, num_epochs=int(5e3), forward_pass_fun=None):
    """
    Forward Pass function should take in xtr, ytr, parameters to calculate loss
    """
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
        loss = forward_pass_fun(Xtr[indices], Ytr[indices], list_parameters)
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
    return list_parameters, list_loss

@torch.no_grad()
def evaluate_model(X, Y, list_parameters, bnmean, bnstd):
    embed = list_parameters[0][X] # shape (num_ngrams[indices], SIZE_CONTEXT-1, SIZE_DIMENSION)
    hidden = torch.einsum('ijk,jkl -> il', embed, list_parameters[1]) #  + list_parameters[2] # hidden states
    hidden_batch = list_parameters[2] * (hidden - bnmean) / bnstd + list_parameters[3]
    h = torch.tanh(hidden_batch) # activated hidden states
    logits = h @ list_parameters[4] + list_parameters[5]
    loss = torch.nn.functional.cross_entropy(logits, Y)

    return loss

bool_demo_train_batch_norm = True
if bool_demo_train_batch_norm:
    print("TRAINING NETWORK - Batch Normalization")
    NUM_EPOCHS = int(5e3) # int(2e4) # int(2e5)
    list_parameters_trained, list_loss = train_model(
        list_parameters_batch_norm, 
        NUM_EPOCHS,
        forward_pass_batch_norm
    )
    plt.plot(list_loss)
    plt.savefig(DIR_OUT / "train-loss-decay-batch-norm.png")
    plt.close()

    with torch.no_grad():
        embed = list_parameters_trained[0][Xtr] # shape (num_ngrams, SIZE_CONTEXT-1, SIZE_DIMENSION)
        hidden = torch.einsum('ijk,jkl -> il', embed, list_parameters_trained[1]) # + list_parameters_trained[2] # hidden states
        bngain_mean = hidden.mean(0,keepdims=True)
        bngain_std = hidden.std(0,keepdims=True)

        print(bngain_mean-bngain_mean_running)
        print(bngain_std-bngain_std_running)

        # Evaluation on the actual calibration
        loss_train = evaluate_model(Xtr, Ytr, list_parameters_trained, bngain_mean, bngain_std)
        loss_dev = evaluate_model(Xdev, Ydev, list_parameters_trained, bngain_mean, bngain_std)
        print("\tTraining loss (actual):", loss_train.item())
        print("\tValidate loss (actual):", loss_dev.item())

        # Evaluation on the estimated calibration
        loss_train = evaluate_model(Xtr, Ytr, list_parameters_trained, bngain_mean_running, bngain_std_running)
        loss_dev = evaluate_model(Xdev, Ydev, list_parameters_trained, bngain_mean_running, bngain_std_running)
        print("\tTraining loss (calibrated):", loss_train.item())
        print("\tValidate loss (calibrated):", loss_dev.item())
