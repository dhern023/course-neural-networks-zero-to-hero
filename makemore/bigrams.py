"""
NGrams, take n-1 tokens to predict the nth token
e.g., bigrams take 2-1 tokens to predict the 2nd token, i.e., P(xi|x_i-1)

Covers the bigrams sections of the video
- Reading in
- Constructing bigrams
- Frequency counting

Tokens are calculated at the character level

NOTE: All functions should normally be at the top,
but they're with their respective sections to make it easier to follow along
NOTE: Things are a little messy since the author jumps around a bit.
"""
import matplotlib.pyplot as plt
import netgraph
import pathlib
import torch
import tqdm

DIRNAME_OUT = "makemore-bigrams"
DIR_READ = pathlib.Path(__file__).resolve().parent
DIR_OUT = pathlib.Path(__file__).resolve().parents[1] / "out" / DIRNAME_OUT
DIR_OUT.mkdir(exist_ok=True, parents=True)

def load_txt_to_list(fname):
    path = pathlib.Path(fname)
    if not path.exists():
        raise f"{fname} not found!"

    list_out = []
    with open(path, "r") as file:
        list_out = file.read().splitlines()

    return list_out

# INSPECTING THE DATASET ==========================================================================

fname = DIR_READ / "names.txt"
words = load_txt_to_list(fname)
print(words[:10])
print(min(words, key = len))
print(max(words, key = len))

# VISUALIZING THE DATASET =========================================================================

# Construct the documents
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
        raise Exception("Smallest token is smaller than context size")

    list_out = []
    for tokens in tqdm.tqdm(list_documents):
        for j in range(0, len(tokens) - size_context + 1):
            vector = construct_vector_ngram(tokens[j:j+size_context], dict_index)
            list_out.append(vector)

    return list_out

def documents_to_indices(documents, dict_token_to_ix):
    """
    Takes tokens -> indices
    """
    return [[dict_token_to_ix[char] for char in doc] for doc in documents]

def extract_ngrams_indices(doc_indices, n):
    """
    Essentially concats all doc indices into one big list,
    then windows them in sizes of n to construct a m x n matrix,
    where m is the total number of n grams
    """
    list_out = []
    for doc in doc_indices:
        list_out.extend(zip(*[doc[i:] for i in range(n)]))
    return list_out

def construct_matrix_adjacency_ngram(list_vectors, dict_index, size_context = 2):
    """
    For the 2D case, A[i][j] is vertex i points to vertex j
    For the 3D case, A[i1][i2][i3] is vertex i1 and i2 point to i3
    Process easily generalizes, but has scaling issues
    """
    out = torch.zeros(tuple([len(dict_index)]*size_context), dtype=torch.int64)
    for vector in list_vectors:
        out[tuple(vector)] += 1

    return out

# Construct the documents
dict_token_to_ix, dict_ix_to_token = construct_map_token_to_index("".join(words))

bool_two_special_chars = False
if bool_two_special_chars:
    list_tokens_extra = ["<S>", "<E>"]
    for token in list_tokens_extra:
        dict_token_to_ix[token] = len(dict_token_to_ix)
        dict_ix_to_token[len(dict_ix_to_token)] = token

    list_documents = [ ["<S>"] + list(string) + ["<E>"] for string in words ]
    tag = "-two-special"
else: # makes this change much later in the video
    list_tokens_extra = ["."]
    for token in list_tokens_extra:
        dict_token_to_ix[token] = len(dict_token_to_ix)
        dict_ix_to_token[len(dict_ix_to_token)] = token

    list_documents = [ ["."] + list(string) + ["."] for string in words ]
    tag = ""

SIZE_NGRAMS = 2
# Convert doc to indices as you go along
# HINT: Use list_documents[0] to just construct the plots for 'emma'
list_vectors = construct_n_grams(list_documents, dict_token_to_ix, size_context=SIZE_NGRAMS)

# versus Convert doc to indices separately
# doc_indices = documents_to_indices(list_documents, dict_token_to_ix)
# list_vectors = extract_ngrams_indices(doc_indices, SIZE_NGRAMS)

matrix_ngrams = torch.tensor(list_vectors, dtype = torch.int64)
N = construct_matrix_adjacency_ngram(matrix_ngrams, dict_token_to_ix, size_context=SIZE_NGRAMS)
N[dict_token_to_ix["a"],[dict_token_to_ix[i] for i in sorted(list(dict_token_to_ix.keys()))]]
print(N)

def plot_adjacency_matrix_graph(matrix, dict_to_tokens, fname_out):
    netgraph.Graph(matrix, arrows=True, node_labels=dict_to_tokens)
    plt.savefig(DIR_OUT / fname_out, dpi=300, bbox_inches='tight')
    plt.close()

def plot_adjacency_matrix_heatmap(matrix, dict_to_tokens, fname_out):
    """
    Plots the square adjacency matrix as a heatmap
    """
    plt.figure(figsize=(16,16))
    plt.imshow(matrix, cmap="Blues")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            label = dict_to_tokens[i] + dict_to_tokens[j]
            plt.text(i,j, label, ha="center", va="bottom", color="gray")
            plt.text(i,j, matrix[i,j], ha="center", va="top", color="gray")
    plt.axis("off")
    plt.savefig(DIR_OUT / fname_out)

fname_graph = f"graph-character{tag}.png"
fname_heatmap = f"heatmap-character{tag}.png"
bool_plot = True
if bool_plot:
    plot_adjacency_matrix_graph(N.numpy(), dict_ix_to_token, fname_out=fname_graph)
    plot_adjacency_matrix_heatmap(N.numpy(), dict_ix_to_token, fname_out=fname_heatmap)

# SAMPLING FROM THE MODEL =========================================================================

g = torch.Generator().manual_seed(2147483647) # default seed from the torch docs
p = torch.rand(3, generator = g)
print(p) # get the same tensor([0.7081, 0.3542, 0.1054])

size_smoothing = 0 # 1
P = (N+size_smoothing) / (N+size_smoothing).sum(dim=SIZE_NGRAMS-1, keepdim=True) # divide by the column-wise sum to normalize N[i,:]
# Double-check these all sum to 1
matrix_ones = P.sum(dim=-1) 
bool_all_ones = torch.allclose(matrix_ones, torch.ones(size = matrix_ones.shape, dtype = matrix_ones.dtype))
print("P is a probability distribution?", bool_all_ones)

def sample_name(matrix, generator, dict_token_to_ix, dict_ix_to_token):
    """
    Normalize the adjacency matrix counts to create probability distributions
        for each index
    These probabilities are passed to a multinomial distribution, 
        which essentially takes a probability distribution and returns an integer
    Continue until we get the stop token
    """
    out = []

    token = "."
    index = dict_token_to_ix[token]
    while True:
        p = matrix[index,:]
        # assert p.sum().item() == 1
        # p = matrix[index]
        index = torch.multinomial(p, num_samples = 1, replacement=True, generator=generator).item()
        out.append(dict_ix_to_token[index])
        if index == dict_token_to_ix[token]:
            break
    
    return "".join(out)

for i in range(7):
    name = sample_name(P, g, dict_token_to_ix, dict_ix_to_token)
    print(name)

# Loss function (negative log likelihood) =========================================================

def calculate_log_likelihood(ngrams, P):
    """
    Get the row of each window based on the indices
    e.g bigrams would be P[ bigrams[0, :], bigrams[1, :] ]
    """
    probs = P[tuple(ngrams.T)]
    log_probs = torch.log(probs)
    return -log_probs.sum().item()

log_likelihood = calculate_log_likelihood(matrix_ngrams, P) 
print(log_likelihood / matrix_ngrams.shape[0])
