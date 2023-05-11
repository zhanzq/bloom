# !/usr/bin/env python
# encoding=utf-8
# author: zhanzq
# email : zhanzhiqiang09@126.com 
# date  : 2023/5/11
#

import torch
from torch.utils.data import DataLoader

from transformers import AutoTokenizer


class LMDataset:
    def __init__(self, file_path, tokenizer, max_seq_length):
        self.file_path = file_path
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.data = []
        self.labels = []
        self.masks = []
        self.load_data()

    def load_data(self):
        with open(self.file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if len(line) > 0:
                    tokens = self.tokenizer.tokenize(line)
                    if len(tokens) > self.max_seq_length - 2:
                        tokens = tokens[:self.max_seq_length - 2]
                    tokens = [self.tokenizer.bos_token] + tokens + [self.tokenizer.eos_token]
                    input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
                    padding_length = self.max_seq_length - len(input_ids)
                    input_ids += [self.tokenizer.pad_token_id] * padding_length
                    self.masks.append([1]*(self.max_seq_length-padding_length) + [0]*padding_length)
                    self.data.append(input_ids)
                    self.labels.append(input_ids[1:] + [self.tokenizer.pad_token_id])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return torch.tensor(self.data[index]), torch.tensor(self.labels[index]), torch.tensor(self.masks[index])


def main():
    model_name_or_path = "/Users/zhanzq/Downloads/pretrained_models/bloom-560m"
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    train_data_path = "../data/train.txt"
    train_dataset = LMDataset(file_path=train_data_path, tokenizer=tokenizer, max_seq_length=20)

    train_data = DataLoader(train_dataset, batch_size=2, shuffle=True)
    for batch in train_data:
        input_ids, labels, masks = batch
        print(input_ids)
        print(labels)
        print(masks)

    return


if __name__ == "__main__":
    main()
