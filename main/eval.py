import config
import engine
import pickle
import data_loader
import numpy as np
import torch
import torch.nn as nn

from encoderDataset import ReferenceDataset
from decoderDataset import TestDataset
from Model import Seq2Seq, Encoder, Decoder
from torch.utils.data import DataLoader

predictions = []

encoder_trainset = ReferenceDataset('test')
encoder_data_loader = DataLoader(encoder_trainset, batch_size=config.BATCH_SIZE, collate_fn=data_loader.create_encoder_batch)

decoder_trainset = TestDataset('test')
decoder_data_loader = DataLoader(decoder_trainset, batch_size=config.BATCH_SIZE, collate_fn=data_loader.create_decoder_batch)

device = torch.device(config.DEVICE)

encoder = Encoder()
encoder.to(device)
decoder = Decoder()
decoder.to(device)

model = Seq2Seq(encoder, decoder)
model.load_state_dict(torch.load(config.MODEL_PATH))
model.to(device)
