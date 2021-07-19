# visualization library
import cv2
from matplotlib import pyplot as plt
# data storing library
import numpy as np
import pandas as pd
# torch libraries
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
#import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader, Dataset, sampler
# architecture and data split library
from sklearn.model_selection import train_test_split
import segmentation_models_pytorch as smp
# augmenation library
from albumentations import (HorizontalFlip, ShiftScaleRotate, Normalize, Resize, Compose, GaussNoise)
from albumentations.pytorch import ToTensor
# others
import os
import pdb
import time
import warnings
import random
from tqdm import tqdm_notebook as tqdm
import concurrent.futures
# warning print supression
warnings.filterwarnings("ignore")

# *****************to reproduce same results fixing the seed and hash*******************
df=pd.read_csv('train_color_img.csv')
# imagenet mean/std will be used as the resnet backbone is trained on imagenet stats
mean, std=(0.485, 0.456, 0.406),(0.229, 0.224, 0.225)


# during traning/val phase make a list of transforms to be used.
# input-->"phase",mean,std
# output-->list
def get_transform(phase,mean,std):
    list_trans=[]
    if phase=='train':
        list_trans.extend([HorizontalFlip(p=0.5)])
    list_trans.extend([Normalize(mean=mean,std=std, p=1), ToTensor()])  #normalizing the data & then converting to tensors
    list_trans=Compose(list_trans)
    return list_trans

'''when dataloader request for samples using index it fetches input image and target mask,
apply transformation and returns it'''
class RGBD_Dataset(Dataset):
    def __init__(self,df,mean,std,phase):
        
        self.fname=df['img'].values.tolist()
        self.mean=mean
        self.std=std
        self.phase=phase

        self.trasnform=get_transform(phase,mean,std)

    def __getitem__(self, idx):
        name=self.fname[idx]

        img_name_path=name
        mask_name_path=img_name_path.split('.')[1].replace('color','truemask')+'.png'
        mask_name_path = '.'+mask_name_path
       
        img=cv2.imread(img_name_path)
        mask=cv2.imread(mask_name_path,cv2.IMREAD_GRAYSCALE)
        
        augmentation=self.trasnform(image=img, mask=mask)

        img_aug=augmentation['image']                           #[3,128,128] type:Tensor
        mask_aug=augmentation['mask']                           #[1,128,128] type:Tensor

        return img_aug, mask_aug

    def __len__(self):
        return len(self.fname)


'''divide data into train and val and return the dataloader depending upon train or val phase.'''
def RGBD_Dataloader(df,mean,std,phase,batch_size):
    df_train, df_valid=train_test_split(df, test_size=0.2, random_state=69)
    df = df_train if phase=='train' else df_valid
    for_loader=RGBD_Dataset(df, mean, std, phase)
    dataloader=DataLoader(for_loader, batch_size=batch_size, pin_memory=True)

    return dataloader

class Trainer(object):
    def __init__(self,model):
        self.num_workers=1
        self.batch_size={'train':10, 'val':10}
        self.accumulation_steps=1#4//self.batch_size['train']
        self.lr=5e-4
        self.num_epochs=1
        self.phases=['train','val']
        self.best_loss=float('inf')
        self.device=torch.device('cpu')#torch.device("cuda:0")
        #torch.set_default_tensor_type("torch.cuda.FloatTensor")
        self.net=model.to(self.device)
        #cudnn.benchmark= True
        self.criterion=torch.nn.BCEWithLogitsLoss()
        self.optimizer=optim.Adam(self.net.parameters(),lr=self.lr)
        self.scheduler=ReduceLROnPlateau(self.optimizer,mode='min',patience=3, verbose=True)
        self.dataloaders={phase: RGBD_Dataloader(df, mean, std, phase=phase,batch_size=self.batch_size[phase],) for phase in self.phases}
        self.losses={phase:[] for phase in self.phases}
        self.dice_score={phase:[] for phase in self.phases}

    def forward(self, inp_images, tar_mask):
        inp_images=inp_images.to(self.device)
        tar_mask=tar_mask.to(self.device)
        pred_mask=self.net(inp_images)
        loss=self.criterion(pred_mask,tar_mask)
        return loss, pred_mask

    def iterate(self, epoch, phase):
        measure=Scores(phase, epoch)
        start=time.strftime("%H:%M:%S")
        print (f"Starting epoch: {epoch} | phase:{phase} | start time:{start}")
        batch_size=self.batch_size[phase]
        self.net.train(phase=="train")
        dataloader=self.dataloaders[phase]
        running_loss=0.0
        total_batches=len(dataloader)
        self.optimizer.zero_grad()

        for itr,batch in enumerate(dataloader):
            start=time.strftime("%H:%M:%S")
            print (f"Epoch: {epoch} | Starting batch: {itr}/{total_batches} | phase:{phase} | start time:{start}")
            images,mask_target=batch
            loss, pred_mask=self.forward(images,mask_target)
            loss=loss/self.accumulation_steps
            if phase=='train':
                loss.backward()
                if (itr+1) % self.accumulation_steps ==0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()
            running_loss+=loss.item()
            pred_mask=pred_mask.detach().cpu()
            measure.update(mask_target,pred_mask)
        epoch_loss=(running_loss*self.accumulation_steps)/total_batches
        #dice=epoch_log(phase, epoch, epoch_loss, measure, start)
        dice=epoch_log(epoch_loss, measure)
        self.losses[phase].append(epoch_loss)
        self.dice_score[phase].append(dice)
        #torch.cuda.empty_cache()
        return epoch_loss

    def start(self):
        for epoch in range (self.num_epochs):
            self.iterate(epoch,"train")
            state = {
                "epoch": epoch,
                "best_loss": self.best_loss,
                "state_dict": self.net.state_dict(),
                "optimizer": self.optimizer.state_dict(),
            }
            with torch.no_grad():
                val_loss=self.iterate(epoch,"val")
                self.scheduler.step(val_loss)
            if val_loss < self.best_loss:
                print("******** New optimal found, saving state ********")
                state["best_loss"] = self.best_loss = val_loss
                torch.save(state, "./model_office.pth")
            print ()

'''calculates dice scores when Scores class for it'''
def dice_score(pred, targs):
    pred = (pred>0).float()
    return 2. * (pred*targs).sum() / (pred+targs).sum()

''' initialize a empty list when Scores is called, append the list with dice scores
for every batch, at the end of epoch calculates mean of the dice scores'''
class Scores:
    def __init__(self, phase, epoch):
        self.base_dice_scores = []

    def update(self, targets, outputs):
        probs = outputs
        dice= dice_score(probs, targets)
        self.base_dice_scores.append(dice)

    def get_metrics(self):
        dice = np.mean(self.base_dice_scores)         
        return dice

'''return dice score for epoch when called'''
def epoch_log(epoch_loss, measure):
    '''logging the metrics at the end of an epoch'''
    dices= measure.get_metrics()    
    dice= dices                       
    print("Loss: %0.4f |dice: %0.4f" % (epoch_loss, dice))
    return dice

model = smp.Unet("resnet18", encoder_weights="imagenet", classes=1, activation=None)
model_trainer = Trainer(model)
model_trainer.start()


'''
#Display some predictions for subset of validation set
test_dataloader=RGBD_Dataloader(df,mean,std,'val',1)

#ckpt_path='/media/shashank/CE7E082A7E080DC1/PycharmProjects/object_detection/model_newloss.pth'
ckpt_path='model_office.pth'

device = torch.device("cpu")#torch.device("cuda")
model = smp.Unet("resnet18", encoder_weights=None, classes=1, activation=None)
model.to(device)
model.eval()
state = torch.load(ckpt_path, map_location=lambda storage, loc: storage)
model.load_state_dict(state["state_dict"])

# start prediction
predictions = []
fig, (ax1,ax2, ax3)=plt.subplots(1,3,figsize=(15,15))
fig.suptitle('predicted_mask//original_mask')
for i, batch in enumerate(test_dataloader):
    if i%10==0:
        images,mask_target = batch
        batch_preds = torch.sigmoid(model(images.to(device)))
        batch_preds = batch_preds.detach().cpu().numpy()

        print(dice_score(batch_preds, mask_target))

        ax1.imshow(np.squeeze(mask_target),cmap='gray')
        ax2.imshow(np.squeeze(batch_preds),cmap='gray')
        ax3.imshow(np.squeeze(batch_preds)>0.5,cmap='gray')
        plt.show()
'''