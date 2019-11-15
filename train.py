import time
import numpy as np
import argparse

import torch
from torch import nn, optim

from preprocess_data import n_vocab, avg_n_jamos_per_segment
from preprocess_data import get_batches
from preprocess_data import train_x, train_y, valid_x, valid_y

from Seq2Seq import Seq2Seq


argsdict = [{"name":'batch_size', "type":int, "default":128, "help":""},
            {"name":'pos_weight', "type":float, "default":1.0, "help":"pos_weight for BCELossWithLogits"},
            {"name":'seq_len', "type":int, "default":200, "help":"sequence length of training data in # of jamos"},
            {"name":'overlap', "type":int, "default":30, "help":"overlap while preprocessing training data in # of jamos"},
            {"name":'n_epochs', "type":int, "default":50, "help":""},
            {"name":'print_every', "type":int, "default":100, "help":""},
            {"name":'initial_lr', "type":float, "default":0.001, "help":"initial learning rate"},
            {"name":'plot_every', "type":int, "default":1, "help":""},
            {"name":'gpu', "type":str, "default":"3", "nargs": '+', "help": 'index of gpu machines to run'},
            {"name":'tf_ratio', "type":float, "default":"1", "help": 'teacher forcing ratio (btwn 0 & 1)'},
            {"name":'load_from', "type":int, "default":0, "help": 'load pre-trained model from this epoch number'}]

parser = argparse.ArgumentParser()

for arg in argsdict:
    parser.add_argument(f'--{arg["name"]}', type=arg['type'], default=arg['default'], help=arg['help'])
    
args = parser.parse_args()


def train(train_x,
          train_y,
          valid_x,
          valid_y,
          model, 
          criterion,
          optimizer,
          initial_lr, 
          batch_size,
          seq_len,
          overlap,
          n_epochs,
          clip=5,
          print_every=100,
          plot_every=1):
    
    torch.backends.cudnn.enabled = False
    
    counter = 0
    model.train()
    train_loss_hist = []
    val_loss_hist = []
    
    for e in range(n_epochs):
        
        optimizer = sqrt_lr_scheduler(optimizer, e + args.load_from, initial_lr) # Decay learning rate 

        for x, y in get_batches(train_x, train_y, batch_size, seq_len, overlap):
             start_time = time.time()
             counter += 1
             
             x, y = torch.from_numpy(x).cuda(), torch.from_numpy(y).cuda()

             optimizer.zero_grad()
             
             pred, train_att_table = model(x, y, tf_ratio=args.tf_ratio) # pred: [64, 2, 200]
             
             loss = criterion(pred, y) 
             
             loss.backward()
             nn.utils.clip_grad_norm_(model.parameters(), clip)
             optimizer.step()
             
             if counter%print_every == 0:
                 val_losses = []
                 model.eval()
                 
                 for val_x, val_y in get_batches(valid_x, valid_y, batch_size, seq_len, overlap):

                     with torch.no_grad():
                         val_x, val_y = torch.from_numpy(val_x).cuda(), torch.from_numpy(val_y).cuda()   
                         val_pred, valid_att_table = model(val_x, val_y, tf_ratio=0)
                         
                         # val_loss doesn't need length_penalty cuz we're not training the model
                         val_loss = criterion(val_pred, val_y) 
                         val_losses.append(val_loss.item())
                 
                 model.train()
                 train_loss_hist.append(loss.item())
                 val_loss_hist.append(np.mean(val_losses))
                 
                 end_time = time.time()
                 epoch_mins, epoch_secs = epoch_time(start_time, end_time)
                 
                 print('%d Epoch: %d/%d  Time: %dm %ds Train Loss: %.8f Valid Loss: %.8f'%(counter/100, e+1+args.load_from, n_epochs, epoch_mins, epoch_secs, loss.item()*100, np.mean(val_losses)*100))
        
        model_name = '_epoch_' + str(e+1+args.load_from) + '.pt'
        print("Saving model to " + model_name + "...")
        torch.save(model.state_dict(), model_name)
        
        if (e+1+args.load_from)%plot_every == 0:
            plot_losses(train_loss_hist, val_loss_hist, name="losses_epoch_"+str(e+1+args.load_from))   
            
            plot_attn(train_att_table[0],  
                      outpath="_train_attn_plot_epoch_" + str(e+1+args.load_from) + ".jpg")
            
            plot_attn(valid_att_table[0],  
                      outpath="_valid_attn_plot_epoch_" + str(e+1+args.load_from) + ".jpg")              
                    
    return train_loss_hist, val_loss_hist
                

      
def length_penalty(pred_batch):  # pred_batch: [batch_size, n_output, seq_len]
    seq_len = pred_batch.size(1)
    avg_n_1s = seq_len/avg_n_jamos_per_segment
    logits_batch = torch.sigmoid(pred_batch)

    penalty = 0
  
    # Give penalty if there are too many 1's (segments TOO SHORT) in each pred of length seq_len:
    penalty = sum([sum(batch.max(dim=0)[1])-avg_n_1s if avg_n_1s<(sum(batch.max(dim=0)[1])*0.6) else 0 for batch in logits_batch])

    # Give penalty if there are too little 1's (segments TOO LONG) in each pred of length seq_len:
    #penalty += sum([avg_n_1s - sum(pred) if avg_n_1s > (sum(pred)*0.6) else 0 for pred in logits_batch])
   
    return penalty/(seq_len*100)


def sqrt_lr_scheduler(optimizer, epoch, initial_lr):
    ''' Decay learning rate by square root of the epoch # '''
    if epoch <= 2: #1: # first 2 epochs
        lr = initial_lr 
    else:
        if epoch%2 == 1:
            #lr = initial_lr / np.sqrt(epoch+2)
            lr = initial_lr / np.sqrt(epoch+1)
        else:
            lr = initial_lr / np.sqrt(epoch)
            
    print("learning rate: ", lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    
    return optimizer


import matplotlib
import matplotlib.pyplot as plt

def plot_losses(train_loss_hist, val_loss_hist, name):
    
    plt.plot(train_loss_hist, label='train loss')
    plt.plot(val_loss_hist, label='valid loss')
    plt.legend(prop={'size': 1})
    plt.ylabel('Loss', fontsize=18)
    plt.xlabel('Counter', fontsize=18)
    plt.savefig("_" + name + '.jpg')
    print("Saving loss graph to _" + name + '.jpg' + "...")

    
def plot_attn(att_weights, encoded_txt=None, outpath=None):
        
    matplotlib.use('Agg')

    fig, ax = plt.subplots()
    cax = ax.matshow(att_weights.cpu().detach().numpy(), aspect='auto', origin='upper',cmap='gray')
    fig.colorbar(cax)
    plt.ylabel('Encoder timestep', fontsize=18)
    plt.xlabel('Decoder timestep', fontsize=18)

    if encoded_txt:
        plt.ylabel('Encoder timestep', fontsize=18)  
        #ax.set_yticklabels([' '] + list(encoded_txt) + [' ']) # Set up axes      
        #ax.yaxis.set_major_locator(ticker.MultipleLocator(1)) # Show label at every tick

    plt.tight_layout()
    plt.savefig(outpath)
    plt.close('all')
    

def epoch_time(start_time, end_time):
    
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    
    return elapsed_mins, elapsed_secs      


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


torch.backends.cudnn.enabled = False
print("Input dim: ", n_vocab)


################################# DEFINE MODEL ################################
model = Seq2Seq(n_vocab = n_vocab, 
                n_embed_text = 128, 
                n_embed_dec = 2, 
                n_hidden_enc = 128, 
                n_hidden_dec = 128, 
                n_layers = 1, 
                n_output = 2, 
                dropout = 0.5)

model = model.cuda()
print(f'The model has {count_parameters(model):,} trainable parameters')
if args.load_from > 0:
    model.load_state_dict(torch.load("_epoch_" + str(args.load_from) + ".pt"))
  

########################### DEFINE LOSS & OPTIMIZER ###########################
positive_label_weights = torch.tensor([args.pos_weight]).cuda()
#criterion = nn.BCEWithLogitsLoss(pos_weight=positive_label_weights)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, amsgrad=True, weight_decay=0)


#################################### TRAIN ####################################
def overlap_data(data, batch_size, seq_len, jump):
    n_batches = int(np.floor(len(data)/(batch_size*seq_len)))   # Truncate Data
    data = data[:n_batches * batch_size * seq_len]
    
    overlaps = []
    for i in range(0, len(data), jump):
        if i+seq_len <= len(data):
            overlaps.append(np.array(data[i:i+seq_len]))
        
    return np.array(overlaps)


train_loss_hist, val_loss_hist = train(train_x,
                                       train_y,
                                       valid_x,
                                       valid_y, 
                                       model, 
                                       criterion, 
                                       optimizer, 
                                       initial_lr = args.initial_lr,
                                       batch_size = args.batch_size,
                                       seq_len = args.seq_len,
                                       overlap = args.overlap,
                                       n_epochs = args.n_epochs,
                                       print_every = args.print_every,
                                       plot_every = args.plot_every)
