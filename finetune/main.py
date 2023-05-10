# !/usr/bin/env python
# encoding=utf-8
# author: zhanzq
# email : zhanzhiqiang09@126.com 
# date  : 2023/4/23
#

import torch
import torch.nn as nn
from transformers import AutoTokenizer, BloomForCausalLM

class BLOOM(nn.Module):
    def __init__(self,
                 model_name_or_path,
                 max_seq_len=512,
                 ):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.model = BloomForCausalLM.from_pretrained(model_name_or_path)
        self.max_seq_len = max_seq_len

    def encoding(self, prompt):
        inputs = self.tokenizer(prompt, return_tensors="pt")
        return inputs

    def generate(self, prompt):
        inputs = self.encoding(prompt)
        batch_size, prompt_len = inputs.shape
        max_step = max(1, self.max_seq_len - prompt_len)
        for _ in range(max_step):
            outputs = self.model(**inputs)
            next_token_id = torch.argmax(outputs.logits, -1)[:, -1:]
            if next_token_id[0,0] == self.tokenizer.eos_token:
                break
            inputs["input_ids"] = torch.cat((inputs["input_ids"], next_token_id), dim=-1)

        tokens = self.tokenizer.convert_ids_to_tokens(next_token_id)
        sequence = self.tokenizer.convert_tokens_to_string(tokens)

        return sequence


def main():
    # "bigscience/bloom-560m"
    model_path_or_flag = "/Users/zhanzq/Downloads/bloom-560m"
    tokenizer = AutoTokenizer.from_pretrained(model_path_or_flag)
    model = BloomForCausalLM.from_pretrained(model_path_or_flag)

    inputs = tokenizer("Hello, my dog is cute.", return_tensors="pt")
    query = "Hello, my dog is cute."

    for _ in range(10):
        inputs = tokenizer(query, return_tensors="pt")

        # inputs.input_ids = torch.cat((inputs.input_ids, next_token_id), -1)
        # inputs.attention_mask = torch.cat((inputs.attention_mask, one_mask), -1)

        outputs = model(**inputs)
        next_token_id = torch.argmax(outputs.logits, -1)[:, -1:]
        torch.argmax(outputs.logits)
        next_token = tokenizer.convert_ids_to_tokens(next_token_id)
        next_word = tokenizer.convert_tokens_to_string(next_token)

        query += next_word
        print(query)

    return


if __name__ == "__main__":
    main()
