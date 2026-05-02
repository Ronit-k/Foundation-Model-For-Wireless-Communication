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

def lwm_inference(preprocessed_chs, input_type, lwm_model, device):
    
    dataset = prepare_for_lwm(preprocessed_chs, device)
    # Process data through LWM
    lwm_loss, embedding_data = evaluate(lwm_model, dataset)
    # print(f'LWM loss: {lwm_loss:.4f}')
    
    if input_type == 'cls_emb':
        embedding_data = embedding_data[:, 0]
    elif input_type == 'channel_emb':  
        embedding_data = embedding_data[:, 1:]
    
    dataset = embedding_data.float()
    return dataset

def prepare_for_lwm(data, device, batch_size=64, shuffle=False):

    input_ids, masked_tokens, masked_pos = zip(*data)
    
    # Store dataset on CPU to avoid CUDA OOM
    input_ids_tensor = torch.tensor(np.array(input_ids), device='cpu').float() 
    masked_tokens_tensor = torch.tensor(np.array(masked_tokens), device='cpu').float() 
    masked_pos_tensor = torch.tensor(np.array(masked_pos), device='cpu').long()

    dataset = TensorDataset(input_ids_tensor, masked_tokens_tensor, masked_pos_tensor)
    
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

def evaluate(model, dataloader):

    model.eval()
    device = next(model.parameters()).device
    running_loss = 0.0
    outputs = []
    criterionMCM = nn.MSELoss()
    
    with torch.no_grad():
        for idx, batch in enumerate(dataloader):
            input_ids = batch[0].to(device)
            masked_tokens = batch[1].to(device)
            masked_pos = batch[2].to(device)

            logits_lm, output = model(input_ids, masked_pos)
            
            output_batch_preproc = output 
            outputs.append(output_batch_preproc.cpu())

            loss_lm = criterionMCM(logits_lm, masked_tokens)
            loss = loss_lm / torch.var(masked_tokens)  
            running_loss += loss.item()
            
    average_loss = running_loss / len(dataloader)
    output_total = torch.cat(outputs, dim=0)
    
    return average_loss, output_total

def create_raw_dataset(data, device):
    """Create a dataset for raw channel data."""
    input_ids, _, _ = zip(*data)
    input_data = torch.tensor(np.array(input_ids), device='cpu')[:, 1:]  
    return input_data.float()
    
