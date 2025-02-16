"""
Downloads the dataset in the video and saves to a text file with the same name
"""

import requests
import pathlib

url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
response = requests.get(url)
assert response.ok
data = response.text

FNAME_OUT = pathlib.Path(__file__).parent / "input.txt"
with open(FNAME_OUT, "w") as file:
    file.write(data)
