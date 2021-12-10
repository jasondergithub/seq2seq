import pickle
from tqdm import tqdm
from encoderDataset import ReferenceDataset
from decoderDataset import TestDataset
import data_loader
from torch.utils.data import DataLoader

def train_fn(src_loader):
    
    for src_data in src_loader:
        
        # src_tokens_tensor = src_data['tokens_tensor']
        # src_segments_tensor = src_data['segments_tensor']
        # src_masks_tensor = src_data['masks_tensor']

        src_tokens_tensor = src_data['tokens_tensor']
        src_segments_tensor = src_data['segments_tensor']
        src_masks_tensor = src_data['masks_tensor']          
     


if __name__ == "__main__":
    
    with open('../table/table'+ str(1) +'.txt', 'rb') as fp:
        table = pickle.load(fp)

    encoder_trainset = ReferenceDataset('train', 1)
    encoder_data_loader = DataLoader(encoder_trainset, batch_size=4, collate_fn=data_loader.create_encoder_batch)

    decoder_trainset = ReferenceDataset('train', 1)
    decoder_data_loader = DataLoader(decoder_trainset, batch_size=4, collate_fn=data_loader.create_decoder_batch)

    train_fn(decoder_data_loader)