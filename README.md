# Overview

Code walkthrough of Andrej Karpathy's Zero-to-Hero Neural Network Tutorial

https://www.youtube.com/playlist?list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ

Idea here is to create reproducible outcomes and build things gradually.
Note: Will skip some things based on personal preference.

# System Requirements

Developed on an i5 5500U CPU via WSL 

# Setup

```
sudo apt-get graphviz
pip3 install requirements.txt
```

# Running

```
python3 -m <folder-name>/filename.py
```

# Troubleshooting
- Torch devices are easy to use but hard to master. Python can raise an erroneous not defined error if you try to assign tensors to other tensors that are not on the same device, i.e., tensor_gpu = tensor_cpu * 9 