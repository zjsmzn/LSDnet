from __future__ import print_function

import numpy as np
from imageio import imsave
import argparse
import time
import torch.utils.data as data_utils

import os
import glob
from PIL import Image
import numpy as np
from torchvision import datasets, transforms
from torchvision import models
from torch.nn import CrossEntropyLoss, DataParallel
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchsummary import summary
from amil_model import Attention
from tensorboardX import SummaryWriter
import argparse
import copy
import json


parser = argparse.ArgumentParser(description='Breakthis data_mynet')
parser.add_argument('--epochs',type=int, default=500, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--reg', type=float, default=10e-5, metavar='R',
                    help='weight decay')
parser.add_argument('--log_dir', type=str, default=None,
                    help='log dir')                  
parser.add_argument('--ckpt_path', type=str, default=None,
                    help='log dir')
parser.add_argument('--device', default='0', type=str, help='comma')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
print( torch.cuda.device_count())


class PatchMethod( torch.utils.data.Dataset ):
    def __init__(self, root='Desktop/screenshots/', mode='train', transform=None):
        self.root = root
        self.mode = mode
        with open(self.root) as f:
            self.raw_samples = json.load(f)
        self.samples = []
        self.transform = transform
        for raw_sample in self.raw_samples:
            label_name = raw_sample[0].split( '/' )[-2]
            if label_name=='lynch':
                label=0
            elif label_name=='non_lynch':
                label=1
            for sample in raw_sample:
                self.samples.append((sample, label))
        if self.mode == 'train':
            random.shuffle( self.samples )

    def __len__(self):
        return len( self.samples )

    def __getitem__(self, index):
        image_dir, label = self.samples[index]
        image_list = glob.glob(os.path.join(image_dir, '*.txt'))
        array = []
        for i, image_path in enumerate(image_list):
            image=np.loadtxt(image_path)
            image=torch.from_numpy(np.array(image).flatten())
            array.append(image)
        array = tuple( array )

        array = torch.stack( array, 0 )
        return (image_dir, array, label)


def train(epoch, model, train_loader, wsi_loss,optimizer, writer):
    model.train()
    train_loss = 0.
    tpr = 0.
    tnr = 0.
    acc = 0.
    p = 0.
    n = 0.
    train_length = len(train_loader)
    for batch_idx, (_, data, label) in enumerate(train_loader):
        bag_label = label[0]
        p += 1 - bag_label
        n += bag_label
        #print(data.shape)
        if args.cuda:
            data, bag_label = data.cuda(), bag_label.cuda()
        data = torch.tensor(data,dtype=torch.float)
        _,y_prob = model(data)
        #print(y_prob)
        loss = wsi_loss(y_prob, bag_label.unsqueeze(0))
        train_loss += loss
        y_prob = F.softmax(y_prob, dim=1)
        y_p, y_hat = torch.max(y_prob, 1)
        #print(y_hat)
        tpr += (y_hat[0] == bag_label & bag_label == 0)
        tnr += (y_hat[0] == bag_label & bag_label == 1)
        acc += (y_hat[0] == bag_label)
        print('Batch_idx : {}/{}, loss : {:.3f}, tpr : {}/{}, tnr : {}/{}, acc : {}/{}'
              .format(batch_idx, train_length, loss.data.cpu(), tpr, p, tnr, n, acc, p+n))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # calculate loss and error for epoch
    train_loss /= train_length
    tpr /= p
    tnr /= n
    acc /= train_length

    writer.add_scalar( 'data/train_acc', acc, epoch )
    writer.add_scalar( 'data/train_TP_rate', tpr, epoch )
    writer.add_scalar( 'data/train_TN_rate', tnr, epoch )
    writer.add_scalar( 'data/train_loss', train_loss, epoch )

    result_train = '{}, Epoch: {}, Loss: {:.4f}, TP rate: {:.4f}, TN rate: {:.4f} Train accuracy: {:.2f}' \
        .format( time.strftime( "%Y-%m-%d %H:%M:%S" ), epoch, train_loss.data.cpu(), tpr, tnr, acc )

    print(result_train)
    return result_train

def val(epoch, model, test_loader, wsi_loss,writer):
    model.eval()
    test_loss = 0.
    tpr = 0.
    tnr = 0.
    acc = 0.
    p = 0.
    n = 0.
    test_length = len(test_loader)
    with torch.no_grad():
        for batch_idx,  (_, data, label) in enumerate(test_loader):
            bag_label = label[0]
            p += 1 - bag_label
            n += bag_label
            data = torch.tensor(data,dtype=torch.float)
            #data=torch.from_numpy(data)
            if args.cuda:
                data, bag_label = data.cuda(), bag_label.cuda()
            _,y_prob = model(data)
            loss = wsi_loss( y_prob, bag_label.unsqueeze(0))
            test_loss += loss
            y_prob = F.softmax(y_prob, dim=1)
            y_p, y_hat = torch.max(y_prob,1)
            tpr += (y_hat[0] == bag_label & bag_label == 0)
            tnr += (y_hat[0] == bag_label & bag_label == 1)
            acc += (y_hat[0] == bag_label)
            print( 'Batch_idx : {}/{}, loss : {:.3f}, tpr : {}/{}, tnr : {}/{}, acc : {}/{}'
                   .format( batch_idx, test_length, loss.data.cpu(), tpr, p, tnr, n, acc, p+n ) )

    test_loss /= test_length
    tpr /= p
    tnr /= n
    acc /= test_length

    writer.add_scalar( 'data/val_acc', acc, epoch )
    writer.add_scalar( 'data/val_TP_rate', tpr, epoch )
    writer.add_scalar( 'data/val_TN_rate', tnr, epoch )
    writer.add_scalar( 'data/val_loss', test_loss, epoch )
    result_test = '{}, Epoch: {}, Loss: {:.4f}, TP rate: {:.4f}, TN rate: {:.4f}, val accuracy: {:.2f}' \
        .format( time.strftime( "%Y-%m-%d %H:%M:%S" ), epoch, test_loss.data.cpu(), tpr, tnr, acc )
    print(result_test)
    return result_test, tpr, tnr

if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn')
    print( 'epoch_{} learning_rate_{}'.format( args.epochs, args.lr ) )
    writer = SummaryWriter( os.path.join( args.log_dir, "epoch" + str( args.epochs ) ) )

    torch.manual_seed( args.seed )
    if args.cuda:
        torch.cuda.manual_seed( args.seed )
        print( '\nGPU is ON!' )

    print( 'Load Train and Test Set' )
    loader_kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

    data_path_train_txt = './data/train_m1_w1.json'
    data_path_val_txt = './data/val_m1_w1.json'

    normalize = transforms.Normalize( mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225] )
    trans = transforms.Compose( [transforms.ToTensor(), normalize] )
    # trans = transforms.Compose([transforms.ToTensor()])
    train_data = PatchMethod( root=data_path_train_txt, transform=trans )
    val_data = PatchMethod( root=data_path_val_txt, mode='test', transform=trans )


    train_loader = torch.utils.data.DataLoader( train_data, shuffle=True, num_workers=4, batch_size=1 )
    val_loader = torch.utils.data.DataLoader( val_data, shuffle=False, num_workers=4, batch_size=1 )

    save_name_txt = os.path.join(args.log_dir, "train_valid_acc.txt")

    os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    if args.ckpt_path:
        print('Loading Model')
        ckpt = torch.load(args.ckpt_path)
        model = Attention()
        model = nn.DataParallel(model,device_ids=None)
        model=model.cuda()
        model.load_state_dict(ckpt['state_dict'])
        optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=args.reg)
        start_epoch = 0
        model_file = open(save_name_txt, "a")
    else:
        print( 'Init Model' )
        model = Attention() 
        model = nn.DataParallel(model,device_ids=None)
        model=model.cuda()
        optimizer = optim.Adam( model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=args.reg )
        start_epoch = 1
        model_file = open(save_name_txt, "w")

    wsi_loss = CrossEntropyLoss().cuda()

    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
    cur_test_acc = 0
    for epoch in range(start_epoch, args.epochs + 1):
        print('----------Start Training----------')
        train_result = train(epoch, model, train_loader, wsi_loss,optimizer, writer)
        torch.save({'epoch': epoch, 'state_dict': model.state_dict()}, os.path.join(args.log_dir, "AMIL_model_latest.ckpt"))
        print('----------Start Testing----------')
        val_result, val_tpr, val_tnr = val(epoch, model, val_loader, wsi_loss,writer)

        model_file.write(train_result + '\n')
        model_file.write(val_result + '\n')
        model_file.flush()
        if  val_tpr+val_tnr > cur_test_acc:
            torch.save( {'epoch': epoch,
                         'state_dict': model.state_dict()},
                        os.path.join(args.log_dir, 'AMIL_model_{}.ckpt'.format(epoch)))
            cur_test_acc = val_tpr+val_tnr
    model_file.close()
