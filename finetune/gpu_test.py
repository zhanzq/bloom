# !/usr/bin/env python
# encoding=utf-8
# author: zhanzq
# email : zhanzhiqiang09@126.com 
# date  : 2023/4/23
#

import sys
import torch
from pynvml import *


def print_gpu_utilization(i=0):
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(i)
    info = nvmlDeviceGetMemoryInfo(handle)
    print(f"GPU memory occupied: {info.used//1024**2} MB.")
    return info.used


def print_summary(result):
    print(f"Time: {result.metrics['train_runtime']:.2f}")
    print(f"Samples/second: {result.metrics['train_samples_per_second']:.2f}")
    print_gpu_utilization()


def use_gpu(i=0):
    torch.ones((1, 1)).to(f"cuda:{i}")
    return print_gpu_utilization(i)


def load_model(model_name, i=0):
    used = print_gpu_utilization(i)
    from transformers import AutoModelForSequenceClassification
    model = AutoModelForSequenceClassification.from_pretrained(f"/workspace/project/models/{model_name}").to(f"cuda:{i}")
    for param in list(model.parameters()):
        print(f"shape: {param.shape}, dtype: {param.dtype}")
    curr = print_gpu_utilization(i)
    print(f"params: {model.num_parameters()//1024**2} M")
    print(f"memory/param: {int((curr-used)/model.num_parameters()*100)/100.0}")


def main(argv):
    model_name = argv[1]
    i = int(argv[2])
    load_model(model_name, i)

    return


if __name__ == "__main__":
    main(sys.argv)
