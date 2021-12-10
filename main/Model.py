import config
import torch
import torch.nn as nn

from transformers import BertModel

class Encoder(nn.Module):
    def __init__(self):
        
        super(Encoder, self).__init__()
        self.bert = BertModel(config.bert_config)
        self.linear = nn.Linear(768, 512)
        self.relu = nn.ReLU(True)
    
    def forward(self, tokens_tensors, segments_tensors, masks_tensors):
        
        outputs = self.bert(input_ids = tokens_tensors, token_type_ids = segments_tensors, 
            attention_mask = masks_tensors)
        
        last_hidden_states = outputs.last_hidden_state
        h_cls = last_hidden_states[0] #only takes CLS as repr. which is [batch, 1, 768]

        out = self.linear(h_cls)
        out = self.relu(out)

        return out

class Decoder(nn.Module):
    def __init__(self):

        super(Decoder, self).__init__()
        self.bert = BertModel(config.bert_config)
        self.linear1 = nn.Linear(768, 512)
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=512, nhead=2, batch_first=True), num_layers=6)
        self.linear2 = nn.Linear(512, 128)
        self.linear3 = nn.Linear(128, 1)
        self.drop = nn.Dropout(0.3)
        self.relu = nn.ReLU(True)

    def forward(self, tokens_tensors, segments_tensors, masks_tensors, hidden):
        
        outputs = self.bert(input_ids = tokens_tensors, token_type_ids = segments_tensors, 
            attention_mask = masks_tensors)
                
        last_hidden_states = outputs.last_hidden_state
        h_cls = last_hidden_states[0] #only takes CLS as repr. which is [batch, 1, 768]

        out = self.linear(h_cls)
        out = self.relu(out)

        decoderOutput = self.decoder(out, hidden)
        decoderOutput = torch.squeeze(decoderOutput, 1) # [batch, 768]
        
        decoderOutput= self.relu(decoderOutput)
        decoderOutput= self.drop(decoderOutput)

        decoderOutput= self.linear2(decoderOutput)
        decoderOutput= self.relu(decoderOutput)
        decoderOutput= self.linear3(decoderOutput)

        return decoderOutput

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, reference_dataloader, test_dataloader):
        pass