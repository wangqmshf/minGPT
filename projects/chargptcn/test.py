"""
Trains a character-level language model.
"""

import os
import sys

import torch
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader

from mingpt.model import GPT
from mingpt.trainer import Trainer
from mingpt.utils import set_seed, setup_logging, CfgNode as CN
from chargpt import get_config
from chargpt import CharDataset

# -----------------------------------------------------------------------------

if __name__ == '__main__':

    # get default config and overrides from the command line, if any
    config = get_config()
    config.merge_from_args(sys.argv[1:])
    print(config)
    setup_logging(config)
    set_seed(config.system.seed)

    # construct the training dataset
    text = open('input.txt', 'r', encoding='utf-8').read() # don't worry we won't run out of file handles
    # print(text)
    train_dataset = CharDataset(config.data, text)
    # construct the trainer object
   
    # construct the model
    config.model.vocab_size = train_dataset.get_vocab_size()
    config.model.block_size = train_dataset.get_block_size()
    model = GPT(config.model)
    trainer = Trainer(config.trainer, model, train_dataset)
    model.load_state_dict(torch.load('./out/chargpt/model.pt'))
    model.eval()
    # sample from the model...
    context = "满纸荒唐言，一把辛酸泪！"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    x = torch.tensor([train_dataset.stoi[s] for s in context], dtype=torch.long)[None,...].to(trainer.device)
    y = model.generate(x, 500, temperature=1.0, do_sample=True, top_k=10)[0]
    completion = ''.join([train_dataset.itos[int(i)] for i in y])
    print(completion)