"""
Character level tokenizer

In practice, subword encodings are the way to go

TODO: Use SentencePiece, TikToken
"""

def character_encode(string, dict_encode):
    """
    Returns iterator of tokens
    """
    token_default = -1

    list_out = []
    for character in string:
        try:
            token = dict_encode[character]
        except KeyError as error:
            token = token_default
        list_out.append(token)

    return list_out    

def character_decode(sequence, dict_decode):
    """
    Returns iterator of characters stored in a string
    """
    default_char = "-"

    list_out = []
    for token in sequence:
        try:
            character = dict_decode[token]
        except KeyError as error:
            character = default_char
        list_out.append(character)
    
    return "".join(list_out)

def construct_character_mappings(string):
    """
    Assumes input contains all data to get representative set

    Returns { token : index } and { index : token }

    Map each token in the vocab to a unique integer,
    which will be its index into the Bag of words vector

    TODO: https://en.wikipedia.org/wiki/Feature_hashing
    NOTE: Fatal flaw is set sorting is random, making debugging a little harder
    """
    vocabulary = sorted(list(set(string))) # sorted by ord(c)

    dict_to_ix = {}
    dict_to_word = {}
    for i, token in enumerate(vocabulary):
        dict_to_ix[token] = i
        dict_to_word[i] = token
    
    return dict_to_ix, dict_to_word

