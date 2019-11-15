import torch
from torch import nn
import torch.nn.functional as F
import random

from .Encoder import Encoder
from .Decoder import Decoder


class Seq2Seq(nn.Module):
    
    def __init__(self, n_vocab, #n_speaker, n_tags, 
                       n_embed_text, #n_embed_speaker, n_embed_tags, 
                       n_embed_dec, 
                       n_hidden_enc, n_hidden_dec, n_layers, 
                       n_output, dropout):
        
        super().__init__()
        
        self.encoder = Encoder(n_vocab=n_vocab, #n_speaker=n_speaker, n_tags=n_tags,
                               n_embed_text=n_embed_text, #n_embed_speaker=n_embed_speaker, n_embed_tags=n_embed_tags,
                               n_hidden_enc=n_hidden_enc, n_layers=n_layers, n_hidden_dec=n_hidden_dec, 
                               dropout=dropout)
        
        self.decoder = Decoder(n_output=n_output, 
                               n_embed=n_embed_dec, 
                               n_hidden_enc=n_hidden_enc, n_hidden_dec=n_hidden_dec, n_layers=n_layers, 
                               dropout=dropout)
        
        
    def forward(self, inputs, targets, tf_ratio=0.5):
        ''' inputs:  [b, input_seq_len(200)]
            targets: [b, input_seq_len(200)]'''
            
        device = inputs.device
        ###########################  1. ENCODER  ##############################
        h = self.encoder.init_hidden(inputs)
        
        last_layer_enc, last_h_enc = self.encoder(inputs, h)              
        
            
        ###########################  2. DECODER  ##############################
        hidden_dec = last_h_enc       #[b, n_layers, n_hidden_dec]
        
        trg_seq_len = targets.size(1)
        
        b = inputs.size(0)
        n_output = self.decoder.n_output
        output = targets[:, 0]
        
        outputs = torch.zeros(b, n_output, trg_seq_len).to(device)

        for t in range(1, trg_seq_len, 1):
            output, hidden_dec, att_weights = self.decoder(output, hidden_dec, last_layer_enc)
            # att_weights : [b, 1, src_seq_len]
            att_weights_table = att_weights if t==1 else torch.cat((att_weights_table, att_weights), dim=1)
            
            outputs[:, :, t] = output #output: [b, n_output]

            if random.random() < tf_ratio:
                output = targets[:, t]
                
            else:
                output = output.max(dim=1)[1]
        # attn_weights_table : [b, trg_seq_len, src_seq_len]
        return outputs, att_weights_table  #[b, n_output, trg_seq_len]
    
