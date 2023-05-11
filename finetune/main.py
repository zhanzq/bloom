# !/usr/bin/env python
# encoding=utf-8
# author: zhanzq
# email : zhanzhiqiang09@126.com 
# date  : 2023/4/23
#

import torch
import torch.nn as nn
from transformers import AutoTokenizer, BloomForCausalLM


class BLOOM:
    def __init__(self,
                 model_name_or_path,
                 train_epochs=3,
                 max_seq_len=512,
                 ):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.model = BloomForCausalLM.from_pretrained(model_name_or_path)
        print("model loading finished!")
        self.max_seq_len = max_seq_len
        self.train_epochs = train_epochs

    def encoding(self, prompt):
        inputs = self.tokenizer(prompt, return_tensors="pt")
        return inputs

    def finetune(self, dataset):
        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=2e-5)
        for _ in range(self.train_epochs):
            for batch in dataset:
                outputs = self.model(batch)
                loss = self.calc_loss(None, None)
                loss.backward()
                optimizer.step()

    def calc_loss(self, pred, label):
        # todo implement
        return None

    def generate(self, prompt):
        self.model.eval()
        inputs = self.encoding(prompt)
        inputs.pop("attention_mask")
        batch_size, prompt_len = inputs["input_ids"].shape
        max_step = max(1, self.max_seq_len - prompt_len)
        for _ in range(max_step):
            outputs = self.model(**inputs)
            next_token_id = torch.argmax(outputs.logits, -1)[:, -1:]
            if next_token_id[0, 0] == self.tokenizer.eos_token:
                break
            inputs["input_ids"] = torch.cat((inputs["input_ids"], next_token_id), dim=-1)

        tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
        sequence = self.tokenizer.convert_tokens_to_string(tokens)

        return sequence


def main():
    # "bigscience/bloom-560m"
    model_name_or_path = "d:\\pretrained_models\\bloom-560m"
    query = "Hello, my dog is cute."
    bloom = BLOOM(model_name_or_path, 100)
    sequence = bloom.generate(prompt=query)
    print(sequence)

    return


if __name__ == "__main__":
    main()
