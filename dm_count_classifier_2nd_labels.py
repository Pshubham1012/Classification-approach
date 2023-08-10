# -*- coding: utf-8 -*-
"""
Created on Wed Feb 22 14:42:42 2023

@author: siddh
"""
import argparse
import torch
import os
import numpy as np
import datasets.crowd as crowd
from models import vgg19
import shutil
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test ')
    parser.add_argument('--device', default='0', help='assign device')
    parser.add_argument('--crop-size', type=int, default=384,
                        help='the crop size of the train image')
    parser.add_argument('--model-path', type=str, default='ckpts\JHU_high_input-384_wot-0.1_wtv-0.01_reg-10.0_nIter-100_normCood-0\best_model_7.pth',
                        help='saved model path')
    parser.add_argument('--data-path', type=str,
                        default='original\Shanghaitech\part_A',
                        help='saved model path')
    parser.add_argument('--dataset', type=str, default='sha',
                        help='dataset name: qnrf, nwpu, sha, shb')
    parser.add_argument('--pred-density-map-path', type=str, default='',
                        help='save predicted density maps when pred-density-map-path is not empty.')
    
    
    args = parser.parse_args()
    
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device  # set vis gpu
    device = torch.device('cuda')
    
    #model_path = args.model_path
    crop_size = args.crop_size
    data_path = args.data_path
    #model_p = 'D:\srp\classification\DM-count master\ckpts/JHU_high_input-384_wot-0.1_wtv-0.01_reg-10.0_nIter-100_normCood-0/best_model_7.pth'
    data_p = 'D:\srp\classification\DM-count master\data\original\Shanghaitech/part_A/validation'
    #print(data_path)
    if args.dataset.lower() == 'qnrf':
        dataset = crowd.Crowd_qnrf(os.path.join(data_path, 'test'), crop_size, 8, method='val')
    elif args.dataset.lower() == 'nwpu':
        dataset = crowd.Crowd_nwpu(os.path.join(data_path, 'val'), crop_size, 8, method='val')
    elif args.dataset.lower() == 'sha' or args.dataset.lower() == 'shb':
        #dataset = crowd.Crowd_sh(os.path.join(data_path, 'test_data'), crop_size, 8, method='val')
        dataset = crowd.Crowd_sh((data_p), crop_size, 8, method='val')
    else:
        raise NotImplementedError
    dataloader = torch.utils.data.DataLoader(dataset, 1, shuffle=False,
                                             num_workers=1, pin_memory=True)
    #print(os.path.join(data_path, 'test_data'))
    if args.pred_density_map_path:
        #import cv2
        if not os.path.exists(args.pred_density_map_path):
            os.makedirs(args.pred_density_map_path)
    
#-----------------------------

    model_path_low = 'D:\srp\classification\DM-count master\dm_count_low/saved_models/low.pth'
    #net_low = CrowdCounter()
    net_low = vgg19()
    net_low.to(device)
    net_low.load_state_dict(torch.load(model_path_low, device))       
    #trained_model_low = os.path.join(model_path_low)
    #network.load_net(trained_model_low, net_low)
    net_low.eval()
    
    model_path_med = 'D:\srp\classification\DM-count master\dm_count_med/saved_models/med.pth'
    #net_med = CrowdCounter()
    net_med = vgg19()
    net_med.to(device)
    net_med.load_state_dict(torch.load(model_path_med, device))      
    #trained_model_med = os.path.join(model_path_med)
    #network.load_net(trained_model_med, net_med)
    net_med.eval()
    
    model_path_high = 'D:\srp\classification\DM-count master\dm_count_high/saved_models/high.pth'
    #net_high = CrowdCounter()
    net_high = vgg19()
    net_high.to(device)
    net_med.load_state_dict(torch.load(model_path_high, device))       
    #trained_model_high = os.path.join(model_path_high)
    #network.load_net(trained_model_high, net_high)
    net_high.eval()
    
    dst_dir_low = r'D:\srp\classification\DM-count master\low'
    dst_dir_med = r'D:\srp\classification\DM-count master\med'
    dst_dir_high = r'D:\srp\classification\DM-count master\high'
    #-----------------------------------------------------------------
    #model = vgg19()
    #model.to(device)
    #model.load_state_dict(torch.load(model_path, device))
    #model.eval()
    #image_errs = []
    im_p = 'D:\srp\classification\DM-count master\data\original\Shanghaitech/part_A/validation/images'
    for inputs, count, name in dataloader:
        image_path = os.path.join(im_p, str(name[0]) + '.jpg')  # for shutil
        inputs = inputs.to(device)
        #print(name,count,inputs)
        assert inputs.size(0) == 1, 'the batch size should equal to 1'
        #------------------------------------low
        with torch.set_grad_enabled(False):
            outputs, _ = net_low(inputs)
            et_count_low = torch.sum(outputs).item()
            
            outputs, _ = net_med(inputs)
            et_count_med = torch.sum(outputs).item()
            
            outputs, _ = net_high(inputs)
            et_count_high = torch.sum(outputs).item()
            
            gt_count = count[0].item()
            #et_count = np.sum(density_map)
            d_low = abs(et_count_low - gt_count)
            d_med = abs(et_count_med - gt_count)
            d_high = abs(et_count_high - gt_count)
            
            mn = min(d_low, d_med, d_high)
            
            if(d_low == mn):
                dst_dir = dst_dir_low
            elif(d_med == mn):
                dst_dir = dst_dir_med
            else:
                dst_dir = dst_dir_high
            shutil.copy(image_path, dst_dir)
        #img_err = count[0].item() - torch.sum(outputs).item()
    
        #print(name, img_err, count[0].item(), torch.sum(outputs).item())
        #image_errs.append(img_err)



#image_errs = np.array(image_errs)
#mse = np.sqrt(np.mean(np.square(image_errs)))
#mae = np.mean(np.abs(image_errs))
#print('{}: mae {}, mse {}\n'.format(model_path, mae, mse))


















