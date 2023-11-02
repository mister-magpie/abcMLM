#%% IMPORTS
# ignore warnings generate by pytorch and lightning
import warnings
warnings.simplefilter("ignore")


# # imports
import lightning.pytorch as pl
import pandas as pd
import numpy as np
import pickle
import inspect
from tqdm.notebook import tqdm

import wandb

from datetime import datetime

from numpy.random import default_rng
import math

import torch
#3090 optimization
torch.set_float32_matmul_precision('medium')

from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

import torchmetrics
from lightning.pytorch import LightningModule, LightningDataModule
from torch.utils.data import random_split

from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import StochasticWeightAveraging, RichProgressBar, RichModelSummary
from lightning.pytorch.strategies import ddp
from lightning.pytorch import Trainer
from pathlib import Path
from datetime import datetime

from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import StochasticWeightAveraging, RichProgressBar, RichModelSummary
from lightning.pytorch.strategies import ddp

import argparse


#%% DATAMODULE
class MLMDataset(Dataset):
    def __init__(self, dataset, block_size, TOKENS, allow_mask_padding=False):
        self.dataset = dataset      
        self.stoi = { tk:i for i,tk in enumerate(TOKENS ) }
        self.itos = { i:tk for i,tk in enumerate(TOKENS ) }
        self.block_size = block_size
        self.vocab_size = len(TOKENS)
        self.IGNORE_TOKEN = -100 # as per pytorch crossentropy default
        self.rng = default_rng()
        self.allow_mask_padding = allow_mask_padding

    def __len__(self):
        return len(self.dataset) 

    def mask_input(self, target, mask_ratio=0.15, ):
        # the mask is as long as the block, but only element before pad get masked
        mask = np.zeros(self.block_size).astype(int)
        seq_len = len(target[ target != self.stoi["<pad>"]])
        if self.allow_mask_padding:
            mask_size = round(mask_ratio * self.block_size)
            if mask_size < 1: mask_size=1
            mask[ self.rng.choice(
                np.arange(0,self.block_size), 
                size = mask_size, 
                replace = False)
                ] = 1
        else:
            mask_size = round(mask_ratio * seq_len)
            if mask_size < 1: mask_size=1
            mask[ self.rng.choice(
                np.arange(0,seq_len), 
                size = mask_size, 
                replace = False)
                ] = 1
            
        # always set mask for EOS so model can learn it better
        mask[target == self.stoi["</s>"]] = 1
        # also adding the first pad, if present
        # if seq_len < 256: mask[seq_len] = 1  
        
        # masking
        input_seq = target.copy()
        input_seq = np.where(mask==1, self.stoi["<mask>"], input_seq)
        # ignore unmasked
        target = np.where(mask==0, self.IGNORE_TOKEN, target)
        
        return input_seq, target        
        

    def __getitem__(self, idx):
        target = np.array([self.stoi[s] for s in self.dataset[idx]])
        # randomly sample from the masking schedule function gamma (See MASKGIT)
        # gamma = np.cos(np.random.uniform(0, np.pi/2))
        gamma = 1 + np.cos(np.random.uniform(0, np.pi/2) + np.pi/2)
        
        input_seq, target = self.mask_input(target, mask_ratio=gamma,)
        input_seq = torch.tensor(input_seq, dtype=torch.long)
        target = torch.tensor(target, dtype=torch.long)

        return input_seq, target

class MLMDataModule(LightningDataModule):
    def __init__(self, data_dir: str = "./datasets/", batch_size: int = 32, max_len=256, allow_mask_padding=False):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size  
        self.max_len = max_len
        self.IGNORE_TOKEN = -100 # as per pytorch crossentropy default
        self.rng = default_rng()
        self.allow_mask_padding = allow_mask_padding

    def load_tokens(self, tokens_path = "./datasets/TOKENS_V4_arranged.pickle"):
        with open(tokens_path,"rb") as f:
            print("Loading tokens:", tokens_path)
            tokens = pickle.load(f)
        tokens = np.append(tokens,'<mask>')
        # put <s> and </s> at the top and <pad> at the bottom
        tokens = np.concatenate([tokens[2:3], tokens[0:2],tokens[3:]])
        tokens = tokens[1:]
        tokens = np.append(tokens,'<pad>')
        return tokens

    def load_dataframe(self, dataset_path = "./datasets/df_v4.pickle"):
        print("Loading dataset:", dataset_path)
        tunes_df = pd.read_pickle(dataset_path).sort_values('length',ascending=False)
        tunes_df["full_abc"] = tunes_df['L'].map(str) + ' ' + tunes_df['M'].map(str) + ' ' + tunes_df['K'].map(str) + ' ' + tunes_df['abc'].map(str)
        return tunes_df[tunes_df.length <= self.max_len-5] # adding <s> L M K </s>

    def create_dataset(self, tunes_df):
            df = tunes_df #[tunes_df.length <= self.max_len-5] # adding <s> L M K </s>    
            strings = '<s> ' + df['L'].map(str) + '\n' + df['M'].map(str) + '\n' + df['K'].map(str) + '\n' + df['abc'].map(str) + ' </s>'
            strings = strings.apply(lambda x: x.split()[:])
            strings = strings.values.reshape(-1,1)
            dataset = np.asarray([self.padding(x) for x in strings[:]])
            return dataset

    #takes a numpy array as input
    def padding(self, array):
        array = array[0]
        array = np.append(array,['<pad>']*(self.max_len-len(array) ))
        assert len(array) == self.max_len
        return np.array(array)
        
    def setup(self, stage):
        print("setting up", stage)
        self.tunes_df = self.load_dataframe()
        self.tokens = self.load_tokens()
        self.stoi = { tk:i for i,tk in enumerate(self.tokens) }
        self.itos = { i:tk for i,tk in enumerate(self.tokens) }
        self.vocab_size = len(self.tokens)
        self.train_df = self.tunes_df.sample(frac=0.9)
        self.test_df = self.tunes_df.drop(self.train_df.index)
        self.train_set = MLMDataset(self.create_dataset(self.train_df), self.max_len, self.tokens, allow_mask_padding=self.allow_mask_padding)
        self.test_set = MLMDataset(self.create_dataset(self.test_df), self.max_len, self.tokens, allow_mask_padding=self.allow_mask_padding)

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size, num_workers=12)

    def val_dataloader(self):
        return DataLoader(self.test_set, batch_size=128, num_workers=12)

#     def test_dataloader(self):
#         return DataLoader(self.mnist_test, batch_size=self.batch_size)

#     def predict_dataloader(self):
#         return DataLoader(self.mnist_predict, batch_size=self.batch_size)

#     def teardown(self, stage: str):
#         # Used to clean-up when the run is finished
#         ...


#%% MODEL
# from pytorch lightning example on transformers
# https://pytorch-lightning.readthedocs.io/en/stable/notebooks/course_UvA-DL/05-transformers-and-MH-attention.html
class CosineWarmupScheduler(optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup, max_iters, min_lr):
        self.warmup = warmup
        self.max_num_iters = max_iters
        self.min_lr = min_lr
        super().__init__(optimizer)

    def get_lr(self):
        lr_factor = self.get_lr_factor(epoch=self.last_epoch)
        lrs = [base_lr * lr_factor for base_lr in self.base_lrs]
        lrs = [lr if (lr >= self.min_lr) else self.min_lr for lr in lrs]
        return lrs

    def get_lr_factor(self, epoch):
        lr_factor = 0.5 * (1 + np.cos(np.pi * epoch / self.max_num_iters))
        if epoch <= self.warmup:
            lr_factor *= epoch * 1.0 / self.warmup
        if epoch > self.max_num_iters:
            return 0.0
        return lr_factor

class MaskedLM(pl.LightningModule):
    def __init__(self, PAD_TOKEN, ntoken, IGNORE_TOKEN=-100, d_model=64, nhead=8, d_hid=128, nlayers=6, dropout=0.0, lr=1e-4, lr_sched=False, max_len=256, weight_decay=1e-2):
        super().__init__()
        self.model_type = 'Transformer'
        
        self.d_model = d_model
        self.PAD_TOKEN = PAD_TOKEN
        self.IGNORE_TOKEN = IGNORE_TOKEN
        self.lr = lr
        self.lr_sched = lr_sched
        self.max_len = max_len
        self.weight_decay = weight_decay
        
        self.embedding = nn.Embedding(ntoken, d_model, padding_idx=PAD_TOKEN)
        self.learned_pos = nn.Embedding(max_len, d_model)

        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_model, 
                nhead=nhead, 
                dim_feedforward=d_hid, 
                dropout=0.0,
                batch_first=True, norm_first=True, 
                activation='gelu'), 
            num_layers=nlayers
        )

        self.ln_f = nn.LayerNorm(d_model)
        self.decoder = nn.Linear(d_model, ntoken,bias=False)
        self.dropout = nn.Dropout(dropout)

        # weight tying see karphaty
        self.embedding.weight = self.decoder.weight
        
        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith('linear2.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * nlayers))

        
        self.save_hyperparameters()
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def configure_optimizers(self, betas=[0.9,0.95]):
            """
            This long function is unfortunately doing something very simple and is being very defensive:
            We are separating out all parameters of the model into two buckets: those that will experience
            weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
            We are then returning the PyTorch optimizer object.
            """

            # separate out all parameters to those that will and won't experience regularizing weight decay
            decay = set()
            no_decay = set()
            whitelist_weight_modules = (nn.Linear, nn.Parameter)
            blacklist_weight_modules = (nn.LayerNorm, torch.nn.Embedding)
            c = 0
            for mn, m in self.named_modules():
                for pn, p in m.named_parameters():
                    c += 1
                    fpn = '%s.%s' % (mn, pn) if mn else pn # full param name
                    # random note: because named_modules and named_parameters are recursive
                    # we will see the same tensors p many many times. but doing it this way
                    # allows us to know which parent module any tensor p belongs to...
                    if fpn.endswith('bias'):
                        # all biases will not be decayed
                        no_decay.add(fpn)
                    elif fpn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                        # weights of blacklist modules will NOT be weight decayed
                        no_decay.add(fpn)
                    elif fpn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                        # weights of whitelist modules will be weight decayed
                        decay.add(fpn)
                    # a bit of an hack to make it work, I'll look into it more
                    elif fpn.endswith('in_proj_weight'):
                        no_decay.add(fpn)

            # subtle: 'transformer.wte.weight' and 'lm_head.weight' are tied, so they
            # will appear in the no_decay and decay sets respectively after the above.
            # In addition, because named_parameters() doesn't return duplicates, it
            # will only return the first occurence, key'd by 'transformer.wte.weight', below.
            # so let's manually remove 'lm_head.weight' from decay set. This will include
            # this tensor into optimization via transformer.wte.weight only, and not decayed.
            decay.remove('decoder.weight')

            # validate that we considered every parameter
            param_dict = {pn: p for pn, p in self.named_parameters()}

            inter_params = decay & no_decay
            union_params = decay | no_decay
            assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
            assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                        % (str(param_dict.keys() - union_params), )

            # create the pytorch optimizer object
            optim_groups = [
                {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": self.weight_decay},
                {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
            ]

            # new PyTorch nightly has a new 'fused' option for AdamW that is much faster
            use_fused = ('fused' in inspect.signature(torch.optim.AdamW).parameters) # and (self.device == 'cuda')
            print(f"using fused AdamW: {use_fused}")
            extra_args = dict(fused=True) if use_fused else dict()
            
            optimizer = torch.optim.AdamW(optim_groups, 
                                          lr=self.lr, 
                                          betas=betas, 
                                          **extra_args
                                          )

            # We don't return the lr scheduler because we need to apply it per iteration, not per epoch
            if self.lr_sched:
                self.lr_scheduler = CosineWarmupScheduler(
                    optimizer, 
                    warmup=150, #self.hparams.warmup, 
                    max_iters=15000,
                    min_lr=1e-4
                )
            
            return optimizer

    def optimizer_step(self, *args, **kwargs):
            super().optimizer_step(*args, **kwargs)
            if self.lr_sched: 
                self.lr_scheduler.step()  # Step per iteration

                
    def forward(self, x):
        attn_mask = (x ==  self.PAD_TOKEN) # True means no attention   
        token_embeddings = self.embedding(x) # each index maps to a (learnable) vector
        
        # from nanogpt
        pos = torch.arange(0, x.shape[1], device=x.device).unsqueeze(0) # shape (1, t)
        position_embeddings = self.learned_pos(pos)
        
        x = self.dropout(token_embeddings + position_embeddings)
        x = self.transformer_encoder(x, src_key_padding_mask = attn_mask)
        x = self.decoder(self.ln_f(x))

        return x
    
    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        o = self.forward(x)
        # ignore tokens are already there
        # y = torch.where(y==self.PAD_TOKEN,self.IGNORE_TOKEN,y)
        loss = F.cross_entropy(
            o.permute(0,2,1), 
            y, 
            ignore_index=-100, 
            reduction="mean", 
            # weight=TOKENS_WEIGHTS.to(self.device)
            )
        
        self.log('train/loss', loss, prog_bar=True, sync_dist=True)
        
        pred = o.argmax(-1)
        acc_top1 = torchmetrics.functional.accuracy(
            o[torch.where(y!=self.IGNORE_TOKEN)], 
            y[torch.where(y!=self.IGNORE_TOKEN)], 
            task="multiclass", 
            num_classes=129, 
            top_k=1
        )
        acc_top5 = torchmetrics.functional.accuracy(
            o[torch.where(y!=self.IGNORE_TOKEN)], 
            y[torch.where(y!=self.IGNORE_TOKEN)], 
            task="multiclass", 
            num_classes=129, 
            top_k=5
        )

        # accuracy = pred[torch.where(y!=self.IGNORE_TOKEN)] == y[torch.where(y!=self.IGNORE_TOKEN)]
        # accuracy = torch.mean(accuracy, dim=-1, dtype=float)
        
        self.log("train/acc/top_1",acc_top1, prog_bar=True, sync_dist=True)
        self.log("train/acc/top_5",acc_top5, prog_bar=True, sync_dist=True)

        if self.lr_sched:
            self.log("lr",self.lr_scheduler.get_last_lr()[0] , prog_bar=True)
        else:
            self.log("lr",self.lr)
        
        return {"loss": loss, "pred": pred, "acc":acc_top1}

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        o = self.forward(x)
        # ignore token already present 
        #y = torch.where(y==self.PAD_TOKEN,self.IGNORE_TOKEN,y)
        loss = F.cross_entropy(
            o.permute(0,2,1), 
            y, 
            ignore_index=-100, 
            # weight=TOKENS_WEIGHTS.to(self.device)
            )
        self.log('valid/loss', loss, prog_bar=True, logger=True, sync_dist=True)
        pred = o.argmax(-1)
        acc_top1 = torchmetrics.functional.accuracy(
            o[torch.where(y!=self.IGNORE_TOKEN)], 
            y[torch.where(y!=self.IGNORE_TOKEN)], 
            task="multiclass", 
            num_classes=129, 
            top_k=1
        )
        acc_top5 = torchmetrics.functional.accuracy(
            o[torch.where(y!=self.IGNORE_TOKEN)], 
            y[torch.where(y!=self.IGNORE_TOKEN)], 
            task="multiclass", 
            num_classes=129, 
            top_k=5
        )

        # accuracy = pred[torch.where(y!=self.IGNORE_TOKEN)] == y[torch.where(y!=self.IGNORE_TOKEN)]
        # accuracy = torch.mean(accuracy, dim=-1, dtype=float)
        
        self.log("valid/acc/top_1",acc_top1, prog_bar=True, sync_dist=True)
        self.log("valid/acc/top_5",acc_top5, prog_bar=True, sync_dist=True)
        return {"loss": loss, "pred": pred, "acc":acc_top1}
    

#%% GENERATION FUNCTIONS
def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    assert logits.dim() == 1  # batch size 1 for now - could be updated for more but the code would be less clear
    top_k = min(top_k, logits.size(-1))  # Safety check
    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value
    return logits

def generate_reels(model, datamodule, n=10, from_structure=True, order="random", verbose=False, top_k=None, top_p=0.9, savepath=None):
    reels = []
    for idx,e in datamodule.test_df[datamodule.test_df["M"] == "M:4/4"].sample(n).iterrows():
        if from_structure:
            input_seq = '<s> ' + e.full_abc + ' </s>' #add SOS and EOS
            input_seq = input_seq.split() 
            input_seq = np.array([ t if (datamodule.stoi[t] < 42 or datamodule.stoi[t] >= 126)  else "<mask>" for t in input_seq ]) # encode non-structural tokens
        else:
            M = ['M:4/4'] #list(tunes_df["M"].sample(1).values)
            K = [np.random.choice(['K:Cmaj','K:Cmin','K:Cmix','K:Cdor'])] # list(tunes_df["K"].sample(1).values)
            input_seq = ["<s>", "L:1/8"] + M + K + ["<mask>"]*e.length  + ["</s>"]
        
        seq_len = len(input_seq)
        # transform sequence to indices
        input_seq = np.array([datamodule.stoi[t] for t in input_seq])
        input_seq = np.pad(input_seq,(0,256-seq_len),"constant", constant_values=datamodule.stoi["<pad>"]).reshape(1,-1) # pad
        masked = list(np.where(input_seq[0]==datamodule.stoi["<mask>"])[0]) # get 
        
        if order == "l2r":
            masked = sorted(masked)
        elif order == "r2l":
            masked = reversed(masked)
        elif order == "random":
            np.random.shuffle(masked) # shuffle the order of random tokens

        while len(masked) != 0:
            i = masked.pop(0)
            # print("prediction index:",i)
            logits = model(
                torch.IntTensor(input_seq), 
            )
            
            if top_k:
                logits[0,i] = top_k_top_p_filtering(logits[0,i], top_k=top_k, filter_value=-torch.inf)
            
            elif top_p:
                logits[0,i] = top_k_top_p_filtering(logits[0,i], top_p=top_p, filter_value=-torch.inf)
                if verbose: print(sum(logits[0,i] > -torch.inf))
            
            probs = torch.nn.functional.softmax(logits,dim=-1)
            
            dist = torch.distributions.Categorical(probs[0,i])
            sampled = dist.sample()
            input_seq[0,i] = sampled
                    
        strout = [datamodule.itos[t] for t in input_seq[0] if t != 128]
        if verbose: print(" ".join(strout).replace("<mask>","_"))
        
        strout.insert(2,"\n")
        strout.insert(4,"\n")
        strout.insert(6,"\n")
        if from_structure:
            strout = "X:{}\nT:based on test {}\n{}".format(str(idx),str(idx),"".join(strout[1:-1]))
        else:
            strout = strout = "X:{}\nT:{}\n{}".format('999'+str(idx),"random","".join(strout[1:-1]))
            
        reels.append(strout)
            
    #save the outputs in a file
    if savepath:
        file = Path(savepath)
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_text("\n\n".join(reels))

    return reels

def generate_jigs(model, datamodule, n=10, from_structure=True, order="random", verbose=False, top_k=None, top_p=0.9, savepath=None):
    jigs = []
    for idx,e in datamodule.test_df[datamodule.test_df["M"] == "M:6/8"].sample(n).iterrows():
        if from_structure:
            input_seq = '<s> ' + e.full_abc + ' </s>' #add SOS and EOS
            input_seq = input_seq.split() 
            input_seq = np.array([ t if (datamodule.stoi[t] < 42 or datamodule.stoi[t] >= 126)  else "<mask>" for t in input_seq ]) # encode non-structural tokens
        else:
            M = ['M:6/8'] #list(tunes_df["M"].sample(1).values)
            K = [np.random.choice(['K:Cmaj','K:Cmin','K:Cmix','K:Cdor'])] # list(tunes_df["K"].sample(1).values)
            input_seq = ["<s>", "L:1/8"] + M + K + ["<mask>"]*e.length  + ["</s>"]
        
        seq_len = len(input_seq)
        # transform sequence to indices
        input_seq = np.array([datamodule.stoi[t] for t in input_seq])
        input_seq = np.pad(input_seq,(0,256-seq_len),"constant", constant_values=datamodule.stoi["<pad>"]).reshape(1,-1) # pad
        masked = list(np.where(input_seq[0]==datamodule.stoi["<mask>"])[0]) # get 

        if order == "l2r":
            masked = sorted(masked)
        elif order == "r2l":
            masked = reversed(masked)
        elif order == "random":
            np.random.shuffle(masked) # shuffle the order of random tokens
            
        while len(masked) != 0:
            i = masked.pop(0)
            # print("prediction index:",i)
            logits = model(
                torch.IntTensor(input_seq), 
            )
            
            if top_k:
                logits[0,i] = top_k_top_p_filtering(logits[0,i], top_k=top_k, filter_value=-torch.inf)
            
            elif top_p:
                logits[0,i] = top_k_top_p_filtering(logits[0,i], top_p=top_p, filter_value=-torch.inf)
                if verbose: print(sum(logits[0,i] > -torch.inf))
            
            probs = torch.nn.functional.softmax(logits,dim=-1)
            
            dist = torch.distributions.Categorical(probs[0,i])
            sampled = dist.sample()
            input_seq[0,i] = sampled
                    
        strout = [datamodule.itos[t] for t in input_seq[0] if t != 128]
        if verbose: print(" ".join(strout).replace("<mask>","_"))
        
        strout.insert(2,"\n")
        strout.insert(4,"\n")
        strout.insert(6,"\n")
        if from_structure:
            strout = "X:{}\nT:based on test {}\n{}".format(str(idx),str(idx),"".join(strout[1:-1]))
        else:
            strout = "X:{}\nT:{}\n{}".format('999'+str(idx),"random","".join(strout[1:-1]))
        jigs.append(strout)
            
    #save the outputs in a file
    if savepath:
        file = Path(savepath)
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_text("\n\n".join(jigs))

    return jigs

def generate_autoregressive(model, prompt, datamodule, n=10, verbose=False, temperature=1.0, top_k=None, top_p=None, savepath=None):
    tunes = []
    
    for idx in range(n):
        if prompt == None:
            M = [np.random.choice(['M:4/4','M:6/8'])]
            K = [np.random.choice(['K:Cmaj','K:Cmin','K:Cmix','K:Cdor'])] 
            input_seq = ["<s>", "L:1/8"] + M + K + ["<mask>"]*256-4
        else:
            input_seq = prompt.split(" ") + ["<mask>"]*(256-len(prompt.split(" ")))
        
        seq_len = len(input_seq)
        assert seq_len == 256
        # transform sequence to indices
        input_seq = np.array([datamodule.stoi[t] for t in input_seq])
        input_seq = np.pad(input_seq,(0,256-seq_len),"constant", constant_values=datamodule.stoi["<pad>"]).reshape(1,-1) # pad
        masked = list(np.where(input_seq[0]==datamodule.stoi["<mask>"])[0]) # get 
        masked = sorted(masked)

        while len(masked) != 0:
            i = masked.pop(0)
            # print("prediction index:",i)
            logits = model(torch.IntTensor(input_seq))
            
            logits = logits[0,i] / temperature
            
            if top_k:
                logits = top_k_top_p_filtering(logits, top_k=top_k, filter_value=-torch.inf)
            
            elif top_p:
                logits = top_k_top_p_filtering(logits, top_p=top_p, filter_value=-torch.inf)
                # if verbose: print(sum(logits[0,i] > -torch.inf))
            
            probs = torch.nn.functional.softmax(logits,dim=-1)
            dist = torch.distributions.Categorical(probs)
            sampled = dist.sample()
            input_seq[0,i] = sampled
            if verbose:
                print( "".join([datamodule.itos[t] for t in input_seq[0]]).replace("<mask>","_"))
            # once we get the end token we exit
            if sampled == datamodule.stoi["</s>"]:
                if verbose: print("eos i:", i)
                break

        strout = [datamodule.itos[t] for t in input_seq[0,:i+1] if t != datamodule.stoi["<pad>"]]
        strout_len = len(strout)
        # if verbose: print("length:", strout_len)
        # if verbose: print(" ".join(strout).replace("<mask>","_"))
        
        strout.insert(2,"\n")
        strout.insert(4,"\n")
        strout.insert(6,"\n")
        
        strout = "X:{}\nT:{}\nN:tokens={}\n{}".format('999'+str(idx),"random autoregressive",strout_len,"".join(strout[1:-1]))
        tunes.append(strout)
            
    #save the outputs in a file
    if savepath:
        file = Path(savepath)
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_text("\n\n".join(tunes))

    return tunes

def fill_masked(model, prompt, datamodule, title="fillmask", verbose=False, temperature=1.0, top_k=None, top_p=None, savepath=None):
 
    input_seq = prompt.split(" ")
    seq_len = len(input_seq)
    
    # transform sequence to indices
    input_seq = np.array([datamodule.stoi[t] for t in input_seq])
    input_seq = np.pad(input_seq,(0,256-seq_len),"constant", constant_values=datamodule.stoi["<pad>"]).reshape(1,-1) # pad
    
    masked = list(np.where(input_seq[0]==datamodule.stoi["<mask>"])[0]) # get 
    masked = sorted(masked)

    while len(masked) != 0:
        i = masked.pop(0)
        # print("prediction index:",i)
        logits = model(torch.IntTensor(input_seq))

        logits = logits[0,i] / temperature

        if top_k:
            logits = top_k_top_p_filtering(logits, top_k=top_k, filter_value=-torch.inf)

        elif top_p:
            logits = top_k_top_p_filtering(logits, top_p=top_p, filter_value=-torch.inf)
            # if verbose: print(sum(logits[0,i] > -torch.inf))

        probs = torch.nn.functional.softmax(logits,dim=-1)
        dist = torch.distributions.Categorical(probs)
        sampled = dist.sample()
        input_seq[0,i] = sampled
        if verbose:
            print( "".join([datamodule.itos[t] for t in input_seq[0] if t != datamodule.stoi["<pad>"] ]).replace("<mask>","_"))
        # once we get the end token we exit
        if sampled == datamodule.stoi["</s>"]:
            if verbose: print("eos i:", i)
            break

    strout = [datamodule.itos[t] for t in input_seq[0] if t != datamodule.stoi["<pad>"]]

    strout.insert(2,"\n")
    strout.insert(4,"\n")
    strout.insert(6,"\n")

    strout = "X:{}\nT:{}\n{}".format(0,title,"".join(strout[1:-1]))
           
    #save the outputs in a file
    if savepath:
        file = Path(savepath)
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_text("\n\n".join(tunes))

    return strout

#%% ARGUMENTS PARSER
argParser = argparse.ArgumentParser()

# model structure
argParser.add_argument("--d_model", help="model size, defines Embedding and Linear Size", type=int, default=256)
argParser.add_argument("--d_hid", help="Hidden size for Attentio Blocks", type=int, default=1024)
argParser.add_argument("--nlayers", help="number of layers", type=int, default=4)
argParser.add_argument("--nhead", help="Number of attention heads in each block. Should perfectly divide d_model", type=int, default=4)
argParser.add_argument("--dropout", help="Dropout Probability", type=float, default=0.0)

# training parameters
argParser.add_argument("-e", "--epochs", help="number of epochs", type=int, default=1)
argParser.add_argument("-b", "--batch_size", help="batch size", type=int, default=256)
argParser.add_argument("--acc", help="accumulate gradients every n batches", type=int, default=1)
argParser.add_argument("--lr", help="learning rate", type=float, default=1e-3)
argParser.add_argument("--lr_decay", help="learning rate decay flag", action="store_true")
argParser.add_argument("--weight_decay", help="weight decay rate", type=float, default=1e-2)
argParser.add_argument("--ckpt", help="specify a checkpoint name", default=None)
argParser.add_argument("--fp", help="floating point precision", default=32)

argParser.add_argument("--resume", help="resume from checkpoint", type=str, default=None)
argParser.add_argument("--mask_pad", help="if true allow masking pad tokens", action="store_true")

argParser.add_argument("--notes", help="add a note the wandb run", type=str, default=None)

# dataset parameters
# argParser.add_argument("--dataset", help="dataset path", default=None)
argParser.add_argument("--max_len", help="maximum sequence length", type=int, default=256)

args = argParser.parse_args()
# print("args=%s" % args)

def train_from_scratch(args):
    if args.ckpt == None:
        timestamp = datetime.now().strftime("%d%h%Y-%H%M%S")
        ckpt_name = "folkMLM_{}".format(timestamp)
        ckpt_path = "./checkpoints/folkMLM_{}.ckpt".format(timestamp)
    else:
        ckpt_name = args.ckpt
        ckpt_path = "./checkpoints/{}.ckpt".format(ckpt_name)
    print("saving checkpoints as %s" % ckpt_name)

    print("-"*50,'\nCreating Dataset')
    datamodule = MLMDataModule(batch_size=args.batch_size, allow_mask_padding=args.mask_pad)
    datamodule.setup(stage="train")

    # model
    print("-"*50,"\nCreating Model")
	
    model = MaskedLM(
        PAD_TOKEN=datamodule.stoi["<pad>"], IGNORE_TOKEN=-100,
        ntoken=datamodule.vocab_size, 
        d_model=args.d_model, d_hid=args.d_hid, nhead=args.nhead, nlayers=args.nlayers,
        dropout=args.dropout, lr=args.lr, lr_sched=args.lr_decay,
        weight_decay=args.weight_decay
        )
    # model = torch.compile(model)    

    # training
    print("-"*50,"\nCreating Trainer")

    wandb_logger = WandbLogger(
        name = ckpt_name,
        project = 'abcMLM', 
        log_model = True,
        notes = args.notes,
        )
    # wandb_logger.experiment.config.update(args.to_dict())
    wandb_logger.watch(model, log_graph=False)

    trainer = Trainer(
        devices = 2,
        accelerator = "auto",
        strategy = ddp.DDPStrategy(find_unused_parameters=False),
        max_epochs = args.epochs,
        # gradient_clip_val=1.0,
        accumulate_grad_batches=1,
        log_every_n_steps=1,
        check_val_every_n_epoch=1,
        logger=wandb_logger,
        precision=args.fp,
        callbacks=[
            RichProgressBar(), 
            RichModelSummary(),
            StochasticWeightAveraging(swa_lrs=args.lr),
            ],
        
    )

    trainer.fit(model, datamodule)
    wandb_logger.experiment.unwatch(model)
    print("Loggin generation examples!")
    # log generation examples
    print("generating strctured reels")
    wandb_logger.log_text(key="reels_struct", columns=["generated tune"], data=np.asarray([generate_reels(model, datamodule, n=10,from_structure=True, savepath="./mlm_outputs/reels_struct_test.abc")]).T)
    print("generating random reels")
    wandb_logger.log_text(key="reels_rand", columns=["generated tune"], data=np.asarray([generate_reels(model, datamodule, n=10,from_structure=False, savepath="./mlm_outputs/reels_rand_test.abc")]).T)
    print("generating strctured jigs")
    wandb_logger.log_text(key="jigs_struct", columns=["generated tune"], data=np.asarray([generate_jigs(model, datamodule, n=10,from_structure=True, savepath="./mlm_outputs/jigs_struct_test.abc")]).T)
    print("generating random jigs")    
    wandb_logger.log_text(key="jigs_rand", columns=["generated tune"], data=np.asarray([generate_jigs(model, datamodule, n=10,from_structure=False, savepath="./mlm_outputs/jigs_rand_test.abc")]).T)
    return model

def resume_training(args):
    print("-"*50,'\nResuming Model:', args.resume)
    run_id = args.resume.split("/")[-1]
    artifcat_ref = 'model-' + run_id + ':latest' #something like "model-eewwhckc:v0"
    # download checkpoint locally (if not already cached)
    run = wandb.init(project = "abcMLM",resume = "must", id = run_id)
    artifact = run.use_artifact(artifcat_ref, type='model')
    artifact_dir = artifact.download()
    # load checkpoint
    model = MaskedLM.load_from_checkpoint(Path(artifact_dir) / "model.ckpt")

    print("-"*50,'\nCreating Dataset')
    datamodule = MLMDataModule(batch_size=args.batch_size, allow_mask_padding=args.mask_pad)
    datamodule.setup(stage="train")

    wandb_logger = WandbLogger(
        name = run.name, 
        project = 'abcMLM', 
        log_model = True, 
        resume="must",)

    wandb_logger.watch(model, log_graph=False)

    trainer = Trainer(
        devices = "auto",
        accelerator = "auto",
        strategy = ddp.DDPStrategy(find_unused_parameters=False),
        max_epochs = args.epochs,
        # gradient_clip_val=1.0,
        accumulate_grad_batches=1,
        log_every_n_steps=1,
        check_val_every_n_epoch=1,
        logger=wandb_logger,
        precision=args.fp,
        callbacks=[
            RichProgressBar(), 
            RichModelSummary(),
            StochasticWeightAveraging(swa_lrs=args.lr),
            ],
    )

    trainer.fit(model,datamodule)
    wandb_logger.experiment.unwatch(model)
    print("Loggin generation examples!")
    # log generation examples
    print("generating strctured reels")
    wandb_logger.log_text(key="reels_struct", columns=["generated tune"], data=np.asarray([generate_reels(model, datamodule, n=10, from_structure=True, savepath="./mlm_outputs/reels_struct_test.abc")]).T)
    print("generating random reels")
    wandb_logger.log_text(key="reels_rand", columns=["generated tune"], data=np.asarray([generate_reels(model, datamodule, n=10, from_structure=False, savepath="./mlm_outputs/reels_rand_test.abc")]).T)
    print("generating strctured jigs")
    wandb_logger.log_text(key="jigs_struct", columns=["generated tune"], data=np.asarray([generate_jigs(model, datamodule, n=10, from_structure=True, savepath="./mlm_outputs/jigs_struct_test.abc")]).T)
    print("generating random jigs")    
    wandb_logger.log_text(key="jigs_rand", columns=["generated tune"], data=np.asarray([generate_jigs(model, datamodule, n=10, from_structure=False, savepath="./mlm_outputs/jigs_rand_test.abc")]).T)
    return model
    # is this necessary?
    run.finish()

#%% MAIN FUNCTION
if __name__ == "__main__":
    print("%s\n FOLK MLM training script\n%s" % ("-"*50,"-"*50 ))
    print(datetime.now().strftime("%d %h %Y - %H:%M:%S"))
    print("-"*50)
    
    if args.resume:
        print("RESUME TRAINING")
        model = resume_training(args)
    
    else:
        print("TRAIN NEW MODEL")
        model = train_from_scratch(args)

    
