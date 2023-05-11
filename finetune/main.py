# !/usr/bin/env python
# encoding=utf-8
# author: zhanzq
# email : zhanzhiqiang09@126.com 
# date  : 2023/4/23
#

import torch
import torch.nn as nn
import os

from torch.utils.data import DataLoader
from transformers import AutoTokenizer, BloomForCausalLM

from data_process.data_process import LMDataset


class BLOOM:
    def __init__(self,
                 model_name_or_path,
                 data_dir="..\\data",
                 train_epochs=3,
                 max_seq_len=512,
                 ):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.model = BloomForCausalLM.from_pretrained(model_name_or_path)
        print("model loading finished!")
        self.max_seq_len = max_seq_len
        self.train_epochs = train_epochs
        train_dataset = LMDataset(os.path.join(data_dir, "train.txt"), tokenizer=self.tokenizer, max_seq_length=40)
        self.train_data = DataLoader(train_dataset, batch_size=4, shuffle=True)
        self.val_data = DataLoader(train_dataset, batch_size=16, shuffle=False)

    def encoding(self, prompt):
        inputs = self.tokenizer(prompt, return_tensors="pt")
        return inputs

    def finetune(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=2e-5)
        criterion = nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(device)
        best_val_loss = float('inf')
        global_step = 0
        total_steps = self.train_epochs * len(self.train_data)
        for epoch in range(self.train_epochs):
            train_loss = 0
            self.model.train()
            for batch in self.train_data:
                global_step += 1
                input_ids, labels, _ = batch
                input_ids = input_ids.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()
                outputs = self.model(input_ids)
                logits = outputs.logits
                loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
                loss.backward()
                optimizer.step()
                batch_loss = loss.item()
                train_loss += batch_loss
                print(f'step {global_step}/{total_steps}, Train Loss: {batch_loss:.4f}')
            train_loss /= len(self.train_data)
            val_loss = 0
            self.auto_test()

            self.model.eval()
            with torch.no_grad():
                for batch in self.val_data:
                    input_ids, labels, _ = batch
                    input_ids = input_ids.to(device)
                    labels = labels.to(device)
                    outputs = self.model(input_ids)
                    logits = outputs.logits
                    loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
                    val_loss += loss.item()
            val_loss /= len(self.val_data)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), 'bloom_model.pt')
            print(f'Epoch {epoch + 1}/{self.train_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

        return

    def generate(self, prompt):
        self.model.eval()
        inputs = self.encoding(prompt)
        inputs.pop("attention_mask")
        batch_size, prompt_len = inputs["input_ids"].shape
        max_step = max(1, self.max_seq_len - prompt_len)
        for _ in range(max_step):
            outputs = self.model(**inputs)
            next_token_id = torch.argmax(outputs.logits, -1)[:, -1:]
            if next_token_id[0, 0] == self.tokenizer.eos_token_id:
                break
            inputs["input_ids"] = torch.cat((inputs["input_ids"], next_token_id), dim=-1)

        tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
        sequence = self.tokenizer.convert_tokens_to_string(tokens)

        return sequence

    def interact_test(self):
        self.model.load_state_dict(torch.load('bloom_model.pt'))
        query = input("User: ")
        while query != "exit":
            sequence = self.generate(prompt=query)
            print(f"Agent: {sequence}")
            query = input("User: ")

    def auto_test(self):
        for query in ["问：你是谁？\n答：", "问：一加一等于几？\n答："]:
            sequence = self.generate(prompt=query)
            print(sequence)

        return


def main():
    # "bigscience/bloom-560m"
    model_name_or_path = "d:\\pretrained_models\\bloom-560m"
    bloom = BLOOM(model_name_or_path, max_seq_len=40, train_epochs=10)
    bloom.interact_test()
    # bloom.finetune()

    return


if __name__ == "__main__":
    main()
