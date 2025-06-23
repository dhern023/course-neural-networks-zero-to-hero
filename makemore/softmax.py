"""
Covers the transition to the neural net by feeding the
one-hot encoded bigrams into a forward-feed neural net with a softmax activation
and a Negative Log-Likelihood loss function

Softmax Function:
Let W be a weight matrix with initial values in N(0,1). If W[i,j] are log counts s.t.
exp(W[i,j]) are counts, then P = exp(W)/sum(exp(W)) := softmax(W)
So softmax yields a probability distribution
"""
import matplotlib.pyplot as plt
import pathlib
import torch
import torch.nn.functional
import tqdm

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

def construct_vector_ngram(tokens_context, dict_index):
    return list(map(lambda w: dict_index[w], tokens_context))

def construct_n_grams(list_documents, dict_index, size_context):
    """
    Constructs the list ngram indices to be used in other functions
    Returns list of list of indices (each list is size size_context)
    """
    if len(min(list_documents, key = len)) < size_context:
        raise("Smallest token is smaller than context size")

    list_out = []
    for tokens in tqdm.tqdm(list_documents):
        for j in range(0, len(tokens) - size_context + 1):
            vector = construct_vector_ngram(tokens[j:j+size_context], dict_index)
            list_out.append(vector)

    return list_out

g = torch.Generator().manual_seed(2147483647) # default seed from the torch docs

DIR_PATH = pathlib.Path(__file__).resolve().parent
fname = DIR_PATH / "names.txt"
words = load_txt_to_list(fname)

# Construct documents
dict_token_to_ix, dict_ix_to_token = construct_map_token_to_index("".join(words))
list_tokens_extra = ["."]
for token in list_tokens_extra:
    dict_token_to_ix[token] = len(dict_token_to_ix)
    dict_ix_to_token[len(dict_ix_to_token)] = token
list_documents = [ ["."] + list(string) + ["."] for string in words ]

# Compute ngrams
SIZE_NGRAMS = 3
list_vectors = construct_n_grams(list_documents, dict_token_to_ix, size_context=SIZE_NGRAMS)
matrix_ngrams = torch.tensor(list_vectors, dtype = torch.int64)

# SOFTMAX =========================================================================================

xs = matrix_ngrams[:,0:SIZE_NGRAMS-1] # 
ys = matrix_ngrams[:,-1] # vector

xenc = torch.nn.functional.one_hot(xs, num_classes = len(dict_token_to_ix)).float()
xenc = xenc.view(-1, len(dict_token_to_ix)*(SIZE_NGRAMS-1))
# plt.imshow(xenc[0])
W = torch.randn(
    size = (len(dict_token_to_ix)*(SIZE_NGRAMS-1), len(dict_token_to_ix)),
    generator=g,
    requires_grad = True
)
# logits = xenc @ W or W[xs, :].squeeze() for n = 2

LEARNING_RATE = 1.0 # 10 50
for i in tqdm.tqdm(range(100)):

    # forward pass
    logits = xenc @ W # treat these as log-counts
    counts = logits.exp() # exp(log-counts) = counts
    probs = counts / counts.sum(dim = 1, keepdim=True) #counts / counts.sum(dim = SIZE_NGRAMS, keepdim=True) # exp(log-counts) / row-sum of exp(log-counts)
    # calculate loss (NLL)
    loss = -probs[torch.arange(ys.shape[0]),ys].log().mean()

    # backward pass
    # initialize gradients
    W.grad = None # more efficient than setting to 0
    loss.backward()
    if i % 10 == 0:
        print(f"Epoch {i} loss: {loss.item()}")

    # update
    W.data += -LEARNING_RATE * W.grad # going against the gradient reduces the loss

print(loss.item())