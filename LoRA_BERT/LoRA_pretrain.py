from datasets import load_dataset
from transformers import DataCollatorWithPadding, BertConfig, BertForSequenceClassification, TrainingArguments, Trainer, BertTokenizerFast
from tokenizers import BertWordPieceTokenizer
import os
import json
import torch
import evaluate
import numpy as np
from sklearn.metrics import accuracy_score
from argparse import ArgumentParser


# if you want to train the tokenizer from scratch (especially if you have custom
# dataset loaded as datasets object), then run this cell to save it as files
# but if you already have your custom data as text files, there is no point using this
def dataset_to_text(dataset, output_filename="data.txt"):
  """Utility function to save dataset text to disk,
  useful for using the texts to train the tokenizer 
  (as the tokenizer accepts files)"""
  with open(output_filename, "w") as f:
    for t in dataset["text"]:
      print(t, file=f)



# Function to compute the metric
def compute_metrics(pred):
  labels = pred.label_ids
  preds = pred.predictions.argmax(-1)
  # calculate accuracy using sklearn's function
  acc = accuracy_score(labels, preds)
  return {
      'accuracy': acc,
  }

def main():
    parser = ArgumentParser()
    #Required Parameters
    parser.add_argument('--input', '-i', default=None, type=str, required=True, help="Input CSV file")
    parser.add_argument('--vocab', '-v', default=100, type=int, help="Vocab size for the tokenizer")
    parser.add_argument('--maxlength', '-m', default=1536, type=int, help="Max sequence length for the model")
    parser.add_argument('--epochs', '-e', default=10, type=int, help="Number of epochs during the training")
    parser.add_argument('--per_train_batch', '-pt', default=None, type=int, required=True, help="Per device train batch size")
    parser.add_argument('--gradient_accumulation_step', '-g', default=None, type=int, required=True, help="Accumulating the gradients before updating the weights")
    parser.add_argument('--per_eval_batch', '-pe', default=None, type=int, required=True, help="Per device eval batch size")
    parser.add_argument('--logging_steps', '-l', default=1000, type=int, help="Evaluate, log and save model checkpoints every chosen step")
    parser.add_argument('--save_steps', '-s', default=1000, type=int, help="Evaluate, log and save model checkpoints every chosen step")

    args = parser.parse_args()
    
    dataset = load_dataset('csv', data_files=f'{args.input}', column_names=['text', 'label'] ,split='train')
    dataset = dataset.shuffle()
    d = dataset.train_test_split(test_size=0.1)
    
    # save the training set to train.txt
    dataset_to_text(d["train"], "train.txt")
    # save the testing set to test.txt
    dataset_to_text(d["test"], "test.txt")
    
    special_tokens = [
      "[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "<S>", "<T>"
    ]
    # if you want to train the tokenizer on both sets
    # files = ["train.txt", "test.txt"]
    # training the tokenizer on the training set
    files = ["train.txt"]
    # 30,522 vocab is BERT's default vocab size, feel free to tweak
    vocab_size = args.vocab
    # maximum sequence length, lowering will result to faster training (when increasing batch size)
    max_length = args.maxlength
    # whether to truncate
    truncate_longer_samples = True
    
    # initialize the WordPiece tokenizer
    tokenizer = BertWordPieceTokenizer()
    # train the tokenizer
    tokenizer.train(files=files, vocab_size=vocab_size, special_tokens=special_tokens)
    # enable truncation up to the maximum 1,536 tokens
    tokenizer.enable_truncation(max_length=max_length)
    
    model_path = "pretrained_bert"
    
    if not os.path.isdir(model_path):
      os.mkdir(model_path)
    
    # save the tokenizer  
    tokenizer.save_model(model_path)
    
    # dumping some of the tokenizer config to config file, 
    # including special tokens, whether to lower case and the maximum sequence length
    with open(os.path.join(model_path, "config.json"), "w") as f:
      tokenizer_cfg = {
          "do_lower_case": True,
          "unk_token": "[UNK]",
          "sep_token": "[SEP]",
          "pad_token": "[PAD]",
          "cls_token": "[CLS]",
          "mask_token": "[MASK]",
          "model_max_length": max_length,
          "max_len": max_length,
      }
      json.dump(tokenizer_cfg, f)
    
    # Commented out IPython magic to ensure Python compatibility.
    # %pip install transformers
    
    #new_tokenizer = BertTokenizerFast(tokenizer_object=tokenizer)
    #new_tokenizer.save_pretrained('./saved_model/')
    
    # when the tokenizer is trained and configured, load it as BertTokenizerFast
    tokenizer = BertTokenizerFast.from_pretrained(model_path)
    
    def encode_without_truncation(exmples):
    	"""Mapping function to tokenize the sentences passed without truncation"""
    	return tokenizer(examples["text"], return_special_tokens_mask=True)
    	
    def encode_with_truncation(examples):
    	"""Mapping function to tokenize the sentences passed with truncation"""
    	return tokenizer(examples["text"], truncation=True, padding="max_length",
		           max_length=max_length, return_special_tokens_mask=True)
		              

    
    # the encode function will depend on the truncate_longer_samples variable
    encode = encode_with_truncation if truncate_longer_samples else encode_without_truncation
    
    # tokenizing the train dataset
    train_dataset = d["train"].map(encode, batched=True)
    # tokenizing the testing dataset
    test_dataset = d["test"].map(encode, batched=True)
    
    if truncate_longer_samples:
      # remove other columns and set input_ids and attention_mask as PyTorch tensors
      train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
      test_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
    else:
      # remove other columns, and remain them as Python lists
      test_dataset.set_format(columns=["input_ids", "attention_mask", "special_tokens_mask", "label"])
      train_dataset.set_format(columns=["input_ids", "attention_mask", "special_tokens_mask", "label"])
     
    # initialize the model with the config
    model_config = BertConfig(vocab_size=vocab_size, max_position_embeddings=max_length)
    model = BertForSequenceClassification(config=model_config).to('cuda')
    #here we can define our special BERT layer architecture!
    #https://huggingface.co/transformers/v3.0.2/model_doc/bert.html#berconfig
    
    
    # Commented out IPython magic to ensure Python compatibility.
    # %pip install accelerate #multiple GPU
    
    # initialize the data collator, for the sequence classification
    data_collator = DataCollatorWithPadding(
        tokenizer=tokenizer
    )
    
    
      
    training_args = TrainingArguments(
        output_dir=model_path,          # output directory to where save model checkpoint
        evaluation_strategy="steps",    # evaluate each `logging_steps` steps
        overwrite_output_dir=True,      
        num_train_epochs=args.epochs,            # number of training epochs, feel free to tweak
        per_device_train_batch_size=args.per_train_batch, # the training batch size, put it as high as your GPU memory fits
        gradient_accumulation_steps=args.gradient_accumulation_step,  # accumulating the gradients before updating the weights
        per_device_eval_batch_size=args.per_eval_batch,  # evaluation batch size
        #auto_find_batch_size=True,      # find the optimal batch size for my GPU
        logging_steps=args.logging_steps,             # evaluate, log and save model checkpoints every 1000 step
        save_steps=args.save_steps,
        load_best_model_at_end=True,  # whether to load the best model (in terms of loss) at the end of training
        save_total_limit=3,           # whether you don't have much space so you let only 3 model weights saved in the disk
    )
    
    
    # initialize the trainer and pass everything to it
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics
        #callbacks=[EarlyStoppingCallback(early_stopping_patience=5)]
    )
    
    trainer.train()

if __name__ =="__main__":
    main()
