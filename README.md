# LoRA_BERT (Natural Language Processing Model for Robust and Accurate Prediction of Long non-coding RNAs)
We introduce a novel pre-trained bidirectional encoder representation called LoRA-BERT. LoRA-BERT is designed to capture the importance of nucleotide-level information during sequence classification, leading to more robust and satisfactory outcomes.

This repository includes several resources: the source code for the LoRA-BERT model, example usages, pre-trained models, fine-tuned models, and etc. We are still actively developing the package and will add more features gradually. 

## Required Packages
Ensure that you have at least one NVIDIA GPU available. We conducted our training using an NVIDIA A100 GPU with 40GB of graphics memory, and the batch size is optimized for this setup. If your GPU has different specifications or memory capacity, you may need to adjust the batch size to suit your hardware.

Make sure to download the following and copy/paste it into the LoRA_BERT directory:

[Datasets](https://drive.google.com/file/d/1bOwe_5-6e0KPkJpKDHdZ1kv1x5NsLIQ9/view?usp=sharing)

[LoRA-BERT](https://drive.google.com/file/d/1cRllgQu3G3qQagj3sLzsa2uOxMWgegS4/view?usp=sharing)


## 1. Data Processing
Please see the template data at `example1.csv`. 

If you are trying to pre-train/fine-tune LoRA-BERT with your own data, please process your data into the same format as it. Note that the sequences are in k-mer representation, so you will need to convert your sequences into that. 

During the process, we removed any sequence less than 100 nts.
Also, the training samples will be around doubled what you have originally collected.   

For example) If you have 500 samples, your training data will around 1,000 samples.

500 seqs(raw sequence -> 3-mer representation) + 500 seqs(raw sequences -> Longest ORF -> 3-mer representation) - number of seqs ess than 100nts

To generate 'example1.csv':

```
python LoRA_input.py --lncRNA {lncRNA.fasta file} --PC {Protein-Coding.fasta file} --output example1.csv
```

## 2. Pre-Training (You can skip if you are using our model)
```
python LoRA_pretrain.py \
    --input {input.csv} \
    --vocab 100 \
    --maxlength 1536 \
    --epochs 10 \
    --per_train_batch 8 \
    --gradient_accumulation_step 8 \
    --per_eval_batch 8 \
    --logging_steps 1000 \
    --save_steps 1000
```
*If you are training with our dataset, you can insert {input.csv} as './Data/Human/Pre-train/train.csv'

## 3. Fine-Tuning (You can skip if you are using our model)
```
python LoRA_finetune.py \
    --input {input.csv} \
    --model_path {trained_model_path} \
    --maxlength 1536 \
    --epochs 10 \
    --per_train_batch 8 \
    --gradient_accumulation_step 8 \
    --per_eval_batch 8 \
    --logging_steps 1000 \
    --save_steps 1000 \
    --weight_decay 0.001 \
    --warmup_step 500
```
*If you are training with our dataset and pre-trained model, you can insert {input.csv} as './Data/Human/Fine_tune/finetune.csv' and {trained_model_path} as './pretrained_bert/Human_pre_model'

## 4. Inference

### 4.1 Full Sequence Inference
The result CSV file will be generated in the results folder.
```
python LoRA_test_full.py --input {input.fasta} --model_path {fine_tuned_model_path} --maxlength 1536
```

### 4.2 Partial Sequence Inference
The result CSV file will be generated in the results folder.
```
python LoRA_test_partial.py --input {input.fasta} --model_path {fine_tuned_model_path} --maxlength 1536
```
*If the sequence doesn't have a start codon('ATG'), it will be removed from the dataset and inference on the remaining sequences.
