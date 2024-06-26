import random
import numpy as np
import torch
import torch.nn as nn
import os
from transformers import BertForSequenceClassification
from transformers import BertTokenizerFast
from datasets import Dataset
import gc
import pickle
from argparse import ArgumentParser
import pandas as pd
from Bio import SeqIO

def seq_kmer(seq, k):
    kmer = [seq[x:x+k] for x in range(len(seq)+1-k)]
    kmers = " ".join(kmer)
    return kmers

def main():

    parser = ArgumentParser()
    parser.add_argument('--input', '-i', default=None, type=str, required=True, help="Input fasta file")
    parser.add_argument('--model_path', '-mp', default=None, type=str, required=True, help="Location of the model")	
    parser.add_argument('--maxlength', '-m', default=1536, type=int, help="Max sequence length for the model")	
    args = parser.parse_args()
    
    max_length = args.maxlength

    A = list(SeqIO.parse(f'{args.input}', 'fasta'))
    A_id = [A[i].id for i in range(len(A))]
    A_seq = [''.join(A[i].seq) for i in range(len(A))]
    A_3mer = [seq_kmer(A_seq[i],3) for i in range(len(A_seq))]
    df = pd.DataFrame({'text':A_3mer})
    dataset = Dataset.from_pandas(df)
    
    model = BertForSequenceClassification.from_pretrained(f'{args.model_path}', num_labels=2).to('cuda')
    tokenizer = BertTokenizerFast.from_pretrained('pretrained_bert', do_lower_case=True)
    
    
    def chunker(seq, batch_size=4):
        return [seq[pos:pos + batch_size] for pos in range(0, len(seq), batch_size)]
        
    total_pre = []
    for sentence_batch in chunker(dataset["text"]):
        inputs = tokenizer(sentence_batch, padding=True, truncation=True, max_length=1536, return_tensors="pt").to('cuda')
        with torch.no_grad():
            outputs = model(input_ids=inputs['input_ids'])
            probs = outputs[0].softmax(1)
            pre1D = probs.cpu().numpy()[:,1].tolist()
            # executing argmax function to get the candidate label
            #candidate = torch.argmax(probs, dim=1)
            total_pre = total_pre + pre1D
    
    df_result = pd.DataFrame({'ID': A_id, 'Prediction':total_pre})
    df_result.to_csv('./results/test_result.csv')
    
if __name__ == "__main__":
    main()
