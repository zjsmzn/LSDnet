import numpy as np
from sklearn.decomposition import PCA
import os
import random
import numpy as np
import argparse

parser = argparse.ArgumentParser(description='Principal Component Analysis')

parser.add_argument('--txt_path', default=None, metavar='TXT_PATH', type=str,
                    help='Path to the input txt file')
parser.add_argument('--save_path', default=None, metavar='SAVE_PATH', type=str,
                    help='Path to the save file')

def run(args):
    data_list=os.listdir(args.txt_path)

    for path in data_list:
        txt_path=args.txt_path+path+'/'
        txt_list=os.listdir(txt_path)
        random.shuffle(txt_list)
        txt_list=txt_list
        wh=[]
        for txt in txt_list:
            f=txt_path+txt
            file1=open(f,'r')
            lines=file1.readlines()
            w=[]
            for line in lines:
                temp=line.split('\n')[0]
                w.append(float(temp))
            wh.append(w)

        wh=np.array(wh).T

        pca = PCA(n_components=256)   #降维
        pca.fit(wh)                  #训练
        newX=pca.fit_transform(wh)   #降维后的数据
        save_path=args.save_path+path+'.txt'
        np.savetxt(save_path,newX)

def main():
    args = parser.parse_args()
    run(args)

if __name__ == '__main__':
    main()