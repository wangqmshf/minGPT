
"""
Trains a GPT to add n-digit numbers.
"""

import os
import sys
import json

import torch
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader

from mingpt.model import GPT
from mingpt.trainer import Trainer
from mingpt.utils import set_seed, setup_logging, CfgNode as CN
from adder import get_config
from adder import AdditionDataset


if __name__ == '__main__':

    # get default config and overrides from the command line, if any
    config = get_config()
    config.merge_from_args(sys.argv[1:])
    print(config)
    setup_logging(config)
    set_seed(config.system.seed)

    # construct train and test datasets
    train_dataset = AdditionDataset(config.data, split='train')
    test_dataset  = AdditionDataset(config.data, split='test')

    # construct the model
    config.model.vocab_size = train_dataset.get_vocab_size()
    config.model.block_size = train_dataset.get_block_size()
    model = GPT(config.model)

    model.load_state_dict(torch.load('./out/adder/model.pt'))
    model.eval()
    # construct the trainer object
    trainer = Trainer(config.trainer, model, train_dataset)

    inp =  torch.tensor([[0,1,3,9]], dtype=torch.long).to(trainer.device)
    n = 3
    with torch.no_grad():
        d1d2d3 = model.generate(inp, n, do_sample=False)
    d3 = d1d2d3[:, -(3):]
    d3 = d3.flip(1) # reverse the digits to their "normal" order
    factors = torch.tensor([[10**i for i in range(2+1)][::-1]]).to(trainer.device)
    # print(factors)
    d3i_pred = (d3 * factors).sum(1)
    print('input sequence  :', inp.tolist())
    print('predicted :', d3i_pred)

