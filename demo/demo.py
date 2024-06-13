import os, sys
base_dir = os.getcwd()
sys.path.insert(0, base_dir)
import model
import importlib
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import cv2, yaml, copy
from easydict import EasyDict as edict
import ctools, gtools
import argparse

# python demo/demo.py -s config/train/config_mpii.yaml -t config/demo/config_mpii.yaml


# 画像ファイルから推定
def main(train, demo):
    # =================================> Setup <=========================
    reader = importlib.import_module("reader." + demo.reader)
    torch.cuda.set_device(demo.device)

    data = demo.data
    load = demo.load
 

    # ===============================> Read Data <=========================
    if data.isFolder: 
        images = os.listdir(data.image)
    else:
        images = [data.image]

    print(f"==> Demo: <==")
    
    logpath = os.path.join(train.save.metapath,
                                train.save.folder, f"{demo.savename}")
    
    logname = demo.logname

  
    if not os.path.exists(logpath):
        os.makedirs(logpath)

    # =============================> Demo <=============================

    net = model.Model()

    statedict = torch.load(
                    load.path, 
                    map_location={f"cuda:{train.device}": f"cuda:{demo.device.device}"}
                )

    net.cuda(); net.load_state_dict(statedict); net.eval()
    
    accs = 0; count = 0


    outfile = open(os.path.join(logpath, logname), 'w')
    outfile.write("name results\n")
    
    data = edict()
    

    with torch.no_grad():
        for j, image in enumerate(images):
            
            if j >= 10:
                break
            
            
            data['face'] = reader.load_image(image)
            
            
            data['face'] = data['face'].cuda()
            
            
            

        
            gazes = net(data)
            

            for k, gaze in enumerate(gazes):

                gaze = gaze.cpu().detach().numpy()
                print(gaze)
                gaze = [str(u) for u in gaze] 
                log = [image] + [",".join(gaze)]
                outfile.write(" ".join(log) + "\n")
    outfile.close()




def main_on_cpu(train, demo):
    # =================================> Setup <=========================
    reader = importlib.import_module("reader." + demo.reader)
    device = torch.device('cpu')  # CPUを使用するように変更

    data = demo.data
    load = demo.load

    # ===============================> Read Data <=========================
    if data.isFolder: 
        images = os.listdir(data.image)
    else:
        images = [data.image]

    print(f"==> Demo: <==")
    
    logpath = os.path.join(train.save.metapath, train.save.folder, f"{demo.savename}")
    logname = demo.logname

    if not os.path.exists(logpath):
        os.makedirs(logpath)

    # =============================> Demo <=============================
    net = model.Model(usegpu=False)
    statedict = torch.load(load.path, map_location=device)  # CPUでモデルを読み込む

    net.load_state_dict(statedict)
    net.to(device)  # CPUにモデルを移動
    net.eval()
    

    outfile = open(os.path.join(logpath, logname), 'w')
    outfile.write("name results\n")
    
    data = edict()

    with torch.no_grad():
        for j, image in enumerate(images):
            if j >= 10:
                break

            data['face'] = reader.load_image(image)
            data['face'] = data['face'].to(device)  # CPUにデータを移動

            gazes = net(data)

            for k, gaze in enumerate(gazes):
                gaze = gaze.cpu().detach().numpy()
                print(gaze)
                gaze = [str(u) for u in gaze] 
                log = [image] + [",".join(gaze)]
                outfile.write(" ".join(log) + "\n")
    outfile.close()



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Pytorch Basic Model Demo')

    parser.add_argument('-s', '--source', type=str,
                        help = 'config path about training')

    parser.add_argument('-t', '--target', type=str,
                        help = 'config path about demo')

    args = parser.parse_args()

    # Read model from train config and Test data in test config.
    train_conf = edict(yaml.load(open(args.source), Loader=yaml.FullLoader))

    demo_conf = edict(yaml.load(open(args.target), Loader=yaml.FullLoader))

    print("=======================>(Begin) Config of training<======================")
    print(ctools.DictDumps(train_conf))
    print("=======================>(End) Config of training<======================")
    print("")
    print("=======================>(Begin) Config for test<======================")
    print(ctools.DictDumps(demo_conf))
    print("=======================>(End) Config for test<======================")
    
    
    if demo_conf.demo.device.usegpu:
        main(train_conf.train, demo_conf.demo)
    else:
        main_on_cpu(train_conf.train, demo_conf.demo)
        