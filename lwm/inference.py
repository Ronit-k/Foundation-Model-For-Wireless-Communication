# -*- coding: utf-8 -*-
"""
Created on Sun Sep 15 18:27:17 2024

@author: salikha4
"""

import os
import csv
import json
import shutil
import random
import argparse
from datetime import datetime
import pandas as pd
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.optim import Adam
import numpy as np
import warnings
warnings.filterwarnings('ignore')

def lwm_inference(preprocessed_chs, input_type, lwm_model, device, batch_size=64, load_data=False):
    """Extract embeddings using the LWM model.
    
    Follows the same pattern as get_lwm_embeddings() in lwm_ca/benchmark.py:
    tensors on CPU, batch-wise GPU transfer, immediate CPU output.
    """
    cache_dir = ".downstream_cache"
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, f"lwm_embeddings_{input_type}.pt")
    
    if load_data and os.path.exists(cache_file):
        print(f"Loading cached embeddings from {cache_file}")
        return torch.load(cache_file)
        
    input_ids, masked_tokens, masked_pos = zip(*preprocessed_chs)
    dataset = TensorDataset(
        torch.tensor(input_ids).float(),
        torch.tensor(masked_tokens).float(),
        torch.tensor(masked_pos).long(),
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    outputs = []
    lwm_model.eval()
    running_loss = 0.0
    criterionMCM = nn.MSELoss()
    
    with torch.no_grad():
        for input_ids_batch, masked_tokens_batch, masked_pos_batch in loader:
            input_ids_batch = input_ids_batch.to(device)
            masked_tokens_batch = masked_tokens_batch.to(device)
            masked_pos_batch = masked_pos_batch.to(device)
            
            logits_lm, output = lwm_model(input_ids_batch, masked_pos_batch)
            outputs.append(output.cpu())
            
            loss_lm = criterionMCM(logits_lm, masked_tokens_batch)
            loss = loss_lm / torch.var(masked_tokens_batch)
            running_loss += loss.item()
    
    average_loss = running_loss / len(loader)
    embedding_data = torch.cat(outputs, dim=0).float()
    
    if input_type == 'cls_emb':
        embedding_data = embedding_data[:, 0]
    elif input_type == 'channel_emb':  
        embedding_data = embedding_data[:, 1:]
        
    if load_data:
        torch.save(embedding_data, cache_file)
        print(f"Saved embeddings to {cache_file}")
    
    return embedding_data

def create_raw_dataset(data, device, load_data=False):
    """Create a dataset for raw channel data."""
    cache_dir = ".downstream_cache"
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, "lwm_raw_data.pt")
    
    if load_data and os.path.exists(cache_file):
        print(f"Loading cached raw data from {cache_file}")
        return torch.load(cache_file)
        
    input_ids, _, _ = zip(*data)
    input_data = torch.tensor(input_ids)[:, 1:]  
    input_data_float = input_data.float()
    
    if load_data:
        torch.save(input_data_float, cache_file)
        print(f"Saved raw data to {cache_file}")
        
    return input_data_float
    
