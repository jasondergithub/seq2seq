import pickle
from tqdm import tqdm
from encoderDataset import ReferenceDataset
from decoderDataset import TestDataset
import data_loader
from torch.utils.data import DataLoader

def loss_fn(outputs, targets):
    return nn.BCEWithLogitsLoss()(outputs, targets)

def train_fn(src_loader, trg_loader, model, optimizer, device, scheduler):
    
    model.train()
    fin_targets = []
    fin_outputs = []
    running_loss = 0    

    for (src_data, trg_data) in tqdm(zip(src_loader, trg_loader)):
        
        src_tokens_tensor = src_data['tokens_tensor']
        src_segments_tensor = src_data['segments_tensor']
        src_masks_tensor = src_data['masks_tensor']
        
        src_tokens_tensor = src_tokens_tensor.to(device, dtype=torch.long)
        src_segments_tensor = src_segments_tensor.to(device, dtype=torch.long)
        src_masks_tensor = src_masks_tensor.to(device, dtype=torch.long)

        reference = [src_tokens_tensor, src_segments_tensor, src_masks_tensor]

        trg_tokens_tensor = trg_data['tokens_tensor']
        trg_segments_tensor = trg_data['segments_tensor']
        trg_masks_tensor = trg_data['masks_tensor']
        trg_label = trg_data['target']

        trg_tokens_tensor = trg_tokens_tensor.to(device, dtype=torch.long)
        trg_segments_tensor =trg_segments_tensor.to(device, dtype=torch.long)
        trg_masks_tensor = trg_masks_tensor.to(device, dtype=torch.long)   
        trg_label_tensor = trg_label.to(device, dtype=torch.float)

        target = [trg_tokens_tensor, trg_segments_tensor, trg_masks_tensor, trg_label_tensor]

        optimizer.zero_grad()
        outputs = model(reference, target)

     


if __name__ == "__main__":
    
    with open('../table/table'+ str(1) +'.txt', 'rb') as fp:
        table = pickle.load(fp)

    encoder_trainset = ReferenceDataset('train', 1)
    encoder_data_loader = DataLoader(encoder_trainset, batch_size=4, collate_fn=data_loader.create_encoder_batch)

    decoder_trainset = TestDataset('train', 1)
    decoder_data_loader = DataLoader(decoder_trainset, batch_size=16, collate_fn=data_loader.create_decoder_batch)

    train_fn(encoder_data_loader, decoder_data_loader)