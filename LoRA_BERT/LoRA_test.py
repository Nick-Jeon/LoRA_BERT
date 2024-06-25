import random
import numpy as np
import torch
import torch.nn as nn
import os
from transformers import BertForSequenceClassification
from transformers import BertTokenizerFast
from datasets import load_dataset
import gc
import pickle
from argparse import ArgumentParser
import pandas as pd


def main():

    parser = ArgumentParser()
    parser.add_argument('--input', '-i', default=None, type=str, required=True, help="Input CSV file")
    parser.add_argument('--model_path', '-mp', default=None, type=str, required=True, help="Location of the model")	
    parser.add_argument('--maxlength', '-m', default=1536, type=int, help="Max sequence length for the model")	
    args = parser.parse_args()
    
    max_length = args.maxlength
    
    dataset = load_dataset('csv', data_files='test.csv', column_names=['text', 'label'] ,split='train')
    target_names = [*set(dataset["label"])]
    
    model = BertForSequenceClassification.from_pretrained(f'{args.model_path}', num_labels=2).to('cuda')
    tokenizer = BertTokenizerFast.from_pretrained('pretrained_bert', do_lower_case=True)
    
    
    def chunker(seq, batch_size=4):
        return [seq[pos:pos + batch_size] for pos in range(0, len(seq), batch_size)]
    
    total_pre = []
    total_pre1D = []
    for sentence_batch in chunker(dataset["text"]):
        inputs = tokenizer(sentence_batch, padding=True, truncation=True, max_length=max_length, return_tensors="pt").to('cuda')
        with torch.no_grad():
            outputs = model(input_ids=inputs['input_ids'])
            probs = outputs[0].softmax(1)
            pre1D = probs.cpu().numpy()[:,1].tolist()
            # executing argmax function to get the candidate label
            candidate = torch.argmax(probs, dim=1)
            predict = np.array(target_names)[candidate.cpu()].tolist()
            #This only require for when input has only one input : change int -> list
            if len(str(predict))==1:
                predict = [predict]
            total_pre=total_pre + predict
            total_pre1D = total_pre1D + pre1D
            
    eval=[]
    for k in range(len(dataset["label"])):
        if dataset["label"][k]==total_pre[k]:
            eval.append(1)
        else:
            eval.append(0)
    score = sum(eval)/len(dataset["label"])
    print('Accuracy:', score)
    
    """
    file = open('predict_label', 'wb')
    # dump information to that file
    pickle.dump(total_pre, file)
    # close the file
    file.close()
    
    file = open('true_label', 'wb')
    # dump information to that file
    pickle.dump(dataset["label"], file)
    # close the file
    file.close()
    
    file = open('roc_predict', 'wb')
    # dump information to that file
    pickle.dump(total_pre1D, file)
    # close the file
    file.close()
    """
    
    df = pd.DataFrame({'True_Label':dataset["label"], 'Pred':total_pre, 'val':total_pre1D})
    df.to_csv('./results/test_result.csv')
    
if __name__ == "__main__":
    main()
