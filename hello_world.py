# --- standard library ---
import os
import math
import random

# --- deep learning ---
import torch
import torch.nn as nn
import torch.optim as optim

# --- external libs ---
import numpy as np
import tiktoken
from tqdm import tqdm
import matplotlib.pyplot as plt

# --- helper functions and variables ---

# --- tokenizer ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

enc = tiktoken.get_encoding("gpt2")

def encode(text: str):
    return enc.encode(text)

def decode(text):
    return enc.decode(text)

# --- embedding modell --- 
vocab_size = 50257 # gpt2-vocab-size
embedding_dim = 128 # vector-size

class EmbeddingModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

    def forward(self,tokens):
        return self.embedding(tokens)

emb_model = EmbeddingModel(vocab_size,embedding_dim)


# --- main function ---
def main():
    text = input("input your question here: ")
    
    tokens = encode(text)
    print("TOKENS: ", tokens)

    # --- create tensor ---
    token_sensor = torch.tensor(tokens).to(device)

    vectors = emb_model(token_sensor)

    print("\nvectors shape:", vectors.shape)

# --- ---
if __name__ == "__main__":
    main()