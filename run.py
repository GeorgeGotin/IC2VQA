import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torchvision
import cv2
from tqdm import tqdm
import seaborn as sns
import pandas as pd
import pickle as PIK
import kornia as K
import torchvision.transforms as T
from torchvision.transforms import v2, InterpolationMode
import itertools
from argparse import ArgumentParser
from torch.utils.data import DataLoader
from PIL import Image
import imageio

# from vsfa_api import VSFA_metric

import Attack

attack_zoo = {
    # 'FGSM':FGSM,
    # 'FGSM-R' : FGSM_R,
    'I2V' : Attack.I2V,
    # 'UAP' : UAP,
    # 'I2V_UAP' : I2V_UAP,
    # 'ENS_I2V' : ENS_I2V,
    # 'I2V_new' : I2V_new
            }


def run_attacks(args):
    arr_eps = sorted(list(args.eps))
    arr_itr = [0] + sorted(list(args.iter))

    image_metrics = args.image_metric
    devices = list(args.device.split('&'))
    attack_name = args.attack_name

    if args.video_metric == 'VSFA':
        vmodel = VSFA_metric(device=devices[0])
    else:
        print('Ooops, no {args.video_metric}')
        exit()

    results = []

    
    data = pd.DataFrame(columns=['attack', 'video_metric', 'baseline', 'video', 'indexes', 'attacked',
       'gain', 'image_metric', 'eps', 'itteration'])
    
    model_names = list(map(lambda x: tuple(map(str.strip, x.split(':'))), image_metrics.split('&')))
    config = {'models' : model_names}

    attack = attack_zoo[attack_name](devices = devices, config=config)

    for video_name in args.video_name:
        print(video_name)
        video = K.color.bgr_to_rgb(ms.video_to_torch('./dataset/videos/'+video_name+'_540_75f.y4m'))
        for indexes in args.indexes:

            if indexes == 'all':
                batch = video
            else:
                left, right = map(int, indexes.split(':'))
                batch = video[left:right]
            bsln = float(vmodel(batch))
            for eps in tqdm(arr_eps):
                attack.eps = eps / 255
                print("Epsilon: ", attack.eps)
                for i in tqdm(range(1, len(arr_itr))):
                    ill = batch.clone()
                    n = arr_itr[i]
                    attack.epoch = n
                    attack.alpha = eps / n
                    res = {}
                    res['attack'] = attack_name
                    res['video_metric'] = 'VSFA'
                    res['baseline'] = bsln
                    #res['baseline_tech'] = bsln['technical']
                    #res['baseline_aest'] = bsln['aesthetic']
                    res['video'] = video_name
                    res['indexes'] = indexes
                    print(f'start attack {device}')
                    ill = itterational_attack(ill, attack, batch_size=args.batch_size)
                    # torchvision.utils.save_image(batch[0], './before_attacked.png')
                    # torchvision.io.write_video('./attacked.mp4', 255*torch.permute(ill, (0,2,3,1)), fps=25)
                    # exit()
                    
                    res['attacked'] = float(vmodel(ill.clone().detach()))
                    #res['image_score_before_attack'] = float(im_score)
                    #attacked_metric = DOVER_from_torch(ill)
                    #res['attacked_tech'] = attacked_metric['technical']
                    #res['attacked_aest'] = attacked_metric['aesthetic']
                    #res['gain_tech'] = res['attacked_tech'] - res['baseline_tech']
                    #res['gain_aest'] = res['attacked_aest'] - res['baseline_aest']
                    #res['attacked'] = float(vs.VSFA_from_torch())
                    res['gain'] = res['attacked'] - res['baseline']
                    res['image_metric'] = attack.public_name
                    res['eps'] = f'{eps}/255'
                    res['itteration'] = arr_itr[i]
                    print(res)
                    results.append(res)
                data = pd.DataFrame.from_dict(results)
                data.to_csv(args.save_file, mode='a', header=False)
                results = []



if __name__ == '__main__':
    parser = ArgumentParser(description='Cross_modal attacks')

    parser.add_argument('--device', default='cuda:0', type=str,
                        help='device (default: cuda:0)')
    parser.add_argument('--video_name', type=str, nargs='*',
                        help='video to attack')    
    parser.add_argument('--image_metric', default='nima', type=str,
                        help='whitebox modal model (default: nima)')
    parser.add_argument('--video_metric', default='vsfa', type=str,
                        help='blackbox modal model (default:  vsfa)')
    parser.add_argument('--attack_name', default='FGSM-I', type=str,
                        help='attack (default: FGSM-I)')
    parser.add_argument('--batch_size', default=30, type=int,
                        help='batch size (default:  30)')
    parser.add_argument('--save_file', default='new_gains.csv', type=str,
                        help='csv file to save results (default: new_gains.csv)')
    parser.add_argument('--layer_name', default=None, type=str,
                        help='layer to be attacked if needed')
    parser.add_argument('--eps', nargs='+', type=int,
                        help='epsilons in form <eps>/255')
    parser.add_argument('--iter', nargs='+', type=int,
                        help='iterations')
    parser.add_argument('--indexes', type=str, default='all', nargs='+',
                        help='indexes of analyze as <start>:<end> or all (default: all)')
    
    args = parser.parse_args()
    print(args)
    run_attacks(args)

    