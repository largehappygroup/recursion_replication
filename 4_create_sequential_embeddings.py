import os
# --- Setting up GPU ---
# os.environ["CUDA_VISIBLE_DEVICES"] = str(get_least_used_gpu())
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

import re
import csv
import glob
import math
import torch
import pynvml
import pickle
# import argparse
import numpy as np
import pandas as pd 
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    AutoModel
)
from tqdm.auto import tqdm
from datasets import load_dataset
import torch.nn.functional as F
from torch.utils.data import DataLoader

############################################################################
####### Setting Environment Variables & Input/Output Paths #################
############################################################################

# --- Checking if GPU is available ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
      
# Load in 4-bit using bitsandbytes
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4"
)

###########################################################################
##################### Functions ###########################################
###########################################################################

def tokenize_for_generation(examples):
        # Assuming the text to embed is in the 'text' column
        return tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length", # Or 'longest' or 'do_not_pad' depending on your needs
            max_length=64, # Adjust max_length as appropriate for your data and model
            return_tensors="pt" # Return PyTorch tensors
        )
        # return inputs

def decide_model(model_name):
    print(model_name)
    try:
        return AutoModelForCausalLM.from_pretrained(model_name, 
                                                    trust_remote_code=True, 
                                                    quantization_config=bnb_config,
                                                    attn_implementation='eager')
    except:
        print(f"Model {model_name} not recognized for embeddings extraction.")
        return None   
        
def generate_embeddings(keystrokes, model, tokenizer):

    input_ids = tokenizer(keystrokes, return_tensors="pt").input_ids.to(device)

    attention_mask = torch.ones_like(input_ids)
    
    hidden_states = {}

    with torch.no_grad():
        outputs = model(
            input_ids = input_ids,
            attention_mask = attention_mask,
            output_hidden_states = True,
            # output_attentions = True,
            return_dict = True
        )
    
    for i, hidden_state_layer in enumerate(outputs.hidden_states):
        layer = hidden_state_layer.squeeze(0).cpu()
        hidden_states[f'layer_{i}'] = layer

    # save hidden states in directories 
    return hidden_states

def extract_summary_token(embedding):
    sm_vector = embedding[-1]
    return sm_vector.to(torch.float32).detach().cpu().numpy()

def z_score_embedding_dictionary(embeddings):
    all_data = torch.stack(list(embeddings.values()), dim=2)
    mean = all_data.mean(dim=2)
    std = all_data.std(dim=2)
    return {k : (v-mean)/std for k,v in embeddings.items()}
        
def process_participants(model_name, model, tokenizer, chosen_layer):
    participant_path = "midprocess"
    participants = os.listdir(participant_path)
    model_outpath = f"output/embeddings"
    
    
    for person in tqdm(participants, desc="Participants"):
        keystroke_bass_path = f"{participant_path}/{person}"
        keystroke_files = glob.glob(f"{keystroke_bass_path}/*keystroke_log.pkl")

        for kf in tqdm(keystroke_files, desc=person, leave=False):
            file_parts = re.split('/', kf)
            scan_block = (file_parts[-1]).removesuffix("_keystroke_log.pkl")

            try:
                with open(kf, 'rb') as f:
                    keystroke_dict = pickle.load(f)
            except:
                continue

            participant_outpath = f"{model_outpath}/{person}"
            if not os.path.exists(participant_outpath):
                os.mkdir(participant_outpath)

            keystroke_embeddings_by_vol = {}
            for vol, keystrokes in keystroke_dict.items():

                if keystrokes == '':
                    continue
                embeddings = generate_embeddings(keystrokes, model, tokenizer)
                zscored_emb = z_score_embedding_dictionary(embeddings)
                summary_embedding = extract_summary_token(zscored_emb[f"layer_{chosen_layer}"]) # performance is similar across layers, so choosing layer 8

                # pull out embeddings layers at these keys
                # then I just want the summary token embeddings
                keystroke_embeddings_by_vol[keystrokes] = summary_embedding
                
            if not os.path.exists(model_outpath):
                os.mkdir(model_outpath)
            with open(f"{participant_outpath}/{scan_block}-keystroke_embeddings.pkl", 'wb') as f:
                pickle.dump(keystroke_embeddings_by_vol, f)
            
        #         break
        #     break
        # break
    

#############################################################################
############# Main Functionality for Creating Embeddings ####################
#############################################################################

def main():
    model_name = 'deepseek_6b'
    model_path = 'deepseek-ai/deepseek-coder-6.7b-base'
    
    # for model_name, model_path in model_names.items():
    print(f"Loading tokenizer and model: {model_name}")
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    tokenizer.pad_token = tokenizer.eos_token  # Ensure pad token is set
    
    model = decide_model(model_path)
    model.to(device) # Move model to appropriate device (e.g., GPU if available)
    model.eval() # Set model to evaluation mode
    print(f"Model on {device}\nTokenizing dataset")
    
    # generating/saving text and embeddings based on prompts
    process_participants(model_name, model, tokenizer, chosen_layer=8)
    
    print(f"Memory usage: {torch.cuda.memory_allocated() / (1024 ** 2):.2f} MB out of {torch.cuda.memory_reserved() / (1024 ** 2):.2f} MB")
    
    # Garbage collection
    del model, tokenizer
    torch.cuda.empty_cache()  # Clear GPU memory
        
        # break


if __name__=="__main__":
    main()
