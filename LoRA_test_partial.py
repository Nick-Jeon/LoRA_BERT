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

# Orf Finder Code
# Four nucleotides of DNA
Nucleotides = ['A', 'C', 'G', 'T']   
# Swap characters
ReverseCompDNA = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C'}

def seq_kmer(seq, k):
    kmer = [seq[x:x+k] for x in range(len(seq)+1-k)]
    kmers = " ".join(kmer)
    return kmers


# Validate DNA sequence in input
def validateSeq(dnaseq):
    tempseq = dnaseq.upper()
    for nuc in tempseq:
        if nuc not in Nucleotides:
            return False
    return tempseq


def reverse_complement(seq):
    """
    reverse_complement function takes the single argument as sequence
    return the reverse+complement ordered.
    """
    # Swapping A with T and G with C vice versa. Reversing newly generated string
    # https://www.geeksforgeeks.org/python-docstrings/
    # https://stackoverflow.com/questions/25188968/reverse-complement-of-dna-strand-using-python
    return ''.join([ReverseCompDNA[nuc] for nuc in seq])[::-1]


# Sequence translation
def translateseq(seq, init_pos=0):
    """
    translateq function takes the arguments
    seq: DNA string sequence
    init_pos: intial position for sequence
    return the sequence i.e. ATGCTGG
    if init_pos=0 ->then start as/from 'ATG'
    if init_pos=1 ->then start as/from '_TGC'
    if init_pos=2 ->then start as/from '__GCT'
    """
    # https://en.m.wikipedia.org/wiki/Complementary_DNA
    # return [seq[pos:pos + 3] for pos in range(init_pos, len(seq) - 2, 3)]
    return ''.join([seq[pos] for pos in range(init_pos, len(seq), 1)])


def gen_reading_frames(seq):
    """
    gen_reading_frames function takes the single argument as sequence
    return the frames list i.e. ['frame', 'frame2',...]
    """
    # Generating eading-frames of sequences including reverse complement
    # including the reverse complement
    # https://www.genome.gov/genetics-glossary/Open-Reading-Frame
    # https://umd.instructure.com/courses/1199394/pages/exam-questions
    # http://justinbois.github.io/bootcamp/2016/lessons/l17_exercise_2.html
    frames = []
    frames.append(translateseq(seq, 0))
    frames.append(translateseq(seq, 1))
    frames.append(translateseq(seq, 2))
    frames.append(translateseq(reverse_complement(seq), 0))
    frames.append(translateseq(reverse_complement(seq), 1))
    frames.append(translateseq(reverse_complement(seq), 2))
    return frames


def searchCodon(sequence, length):
    """
    searchCodon function takes two arguments:
    sequence: DNA sequence string
    length: Used to display the ORF above the length(LIMIT)

    return the ORF
    """
    MAXLIMIT = length

    STARTcodon = "ATG"
    STOPcodon = ["TAA", "TAG", "TGA"]

    position = 0
    ORFsequence = ""
    aasequence = ""
    # dictionary of the amino acids according to corresponding DNA string
    gencode_table = {
        'ATA': 'I', 'ATC': 'I', 'ATT': 'I', 'ATG': 'M',
        'ACA': 'T', 'ACC': 'T', 'ACG': 'T', 'ACT': 'T',
        'AAC': 'N', 'AAT': 'N', 'AAA': 'K', 'AAG': 'K',
        'AGC': 'S', 'AGT': 'S', 'AGA': 'R', 'AGG': 'R',
        'CTA': 'L', 'CTC': 'L', 'CTG': 'L', 'CTT': 'L',
        'CCA': 'P', 'CCC': 'P', 'CCG': 'P', 'CCT': 'P',
        'CAC': 'H', 'CAT': 'H', 'CAA': 'Q', 'CAG': 'Q',
        'CGA': 'R', 'CGC': 'R', 'CGG': 'R', 'CGT': 'R',
        'GTA': 'V', 'GTC': 'V', 'GTG': 'V', 'GTT': 'V',
        'GCA': 'A', 'GCC': 'A', 'GCG': 'A', 'GCT': 'A',
        'GAC': 'D', 'GAT': 'D', 'GAA': 'E', 'GAG': 'E',
        'GGA': 'G', 'GGC': 'G', 'GGG': 'G', 'GGT': 'G',
        'TCA': 'S', 'TCC': 'S', 'TCG': 'S', 'TCT': 'S',
        'TTC': 'F', 'TTT': 'F', 'TTA': 'L', 'TTG': 'L',
        'TAC': 'Y', 'TAT': 'Y', 'TAA': '_', 'TAG': '_',
        'TGC': 'C', 'TGT': 'C', 'TGA': '_', 'TGG': 'W'
    }

    maxORF = list()

    while (position < len(sequence) - 2):
        # currentCodon-> take 3 steps.
        # i.e. ATGCTGG-> 'ATG'->'TGC'->'GCT'->...
        currentCodon = sequence[position:position + 3]

        # currentCodon is True
        # currentCodon->codon='ATG'
        if (currentCodon == STARTcodon):  # ->Start codon
            # Run until currentCodon is not in STOPcodon and gencode table
            while not (currentCodon in STOPcodon) and (currentCodon in gencode_table):
                ORFsequence = ORFsequence + currentCodon
                # Replace currentCodon i.e. 'TCA' from gencode table
                # Concatenate string with previous assequence(Amino Acid Seq)
                aasequence += gencode_table[currentCodon]
                # Postion for codon to move forward to select next codon
                position = position + 3
                # Replace currentCodon position
                currentCodon = sequence[position:position + 3]
            # Concatenate ORFs with respect to currentCodon position
            ORFsequence = ORFsequence + currentCodon
            # ->Stop codon

            # Check ORFsequence length
            if (len(ORFsequence) > MAXLIMIT):
                #print('\n>>>START CODON')
                #print(f'{ORFsequence}')
                maxORF.append(ORFsequence)
                #print('<<<END CODON')
                #print("\nAmino Acid Sequence:")
                #print(f'{aasequence} Lenth: {len(aasequence)}')
                # Reset ORFsequence and assequence
            ORFsequence = ""
            aasequence = ""
        # Increase postion
        position = position + 3
    return maxORF

def longest_orf(seq1):

    if seq1.find('ATG')==-1:
        return
        
    #print()
    #inputfile = input('Enter DNA String: ')
    genome = validateSeq(seq1)

    ORFList = list()
    iteration = 0
    #print('\nReading Frames')
    # printing all the 6 frames and their all orfs now
    for frame in gen_reading_frames(genome):
        #print(f'\n***************[FRAME {iteration + 1}]***************')
        # return ORF on each iteration
        ORF = searchCodon(frame, 0)
        # Add ORF in ORFList on each iteration
        ORFList.append(ORF)
        iteration += 1

    # Now finding the longest orf among the 6 frames, if you dont need the longest orf please comment the code below

    #print(f'\n\n*************LONGEST ORF*************')
    #print('The longest ORF among all as below\n')
    # Converting ORFList into Sting
    s = str(ORFList)
    # Delimitier
    delimiter = ","
    string = s.split(delimiter)  # Converts the string into list
    # Finding the Longest ORF from the list
    #print("longest ORF sequence is = ", max(string, key=len))
    #print("This code is Published by Hassan Raza on Github")

    B_lnc = max(string, key=len)
    for i in range(len(B_lnc)):
        if B_lnc[i] != 'A' and B_lnc[i] != 'C' and B_lnc[i] != 'G' and B_lnc[i] != 'T':
            continue
        else:
            strt = i
            break
    for i in range(len(B_lnc)):
        if B_lnc[-i] != 'A' and B_lnc[-i] != 'C' and B_lnc[-i] != 'G' and B_lnc[-i] != 'T':
            continue
        else:
            endcodon = -i+1
            break
    return B_lnc[strt:endcodon]


def main():

    parser = ArgumentParser()
    parser.add_argument('--input', '-i', default=None, type=str, required=True, help="Input fasta file")
    parser.add_argument('--model_path', '-mp', default=None, type=str, required=True, help="Location of the model")	
    parser.add_argument('--maxlength', '-m', default=1536, type=int, help="Max sequence length for the model")	
    args = parser.parse_args()
    
    max_length = args.maxlength

    # Loading the Fasta data samples
    A = list(SeqIO.parse(f'{args.input}', 'fasta'))
    A_seq = [''.join(A[i].seq) for i in range(len(A))]
    # Finding the longest ORF
    A_orf = [longest_orf(A_seq[i]) for i in range(len(A_seq))]
    A_orf_3mer = []
    A_id = []
    for i in range(len(A_orf)):
        if A_orf[i]!=None:
            A_orf_3mer.append(seq_kmer(A_orf[i],3))
            A_id.append(A[i].id)

    df = pd.DataFrame({'text':A_orf_3mer})
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
