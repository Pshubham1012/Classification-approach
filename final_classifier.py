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
from torchvision import transforms
from PIL import Image
import io
from src import utils
if __name__ == '__main__':
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False
    vis = False
    save_output = True
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
    class_model1 = torch.load(r'D:\srp\classification\DM-count master\classification_models/model1.pt')
    class_model2 = torch.load(r'D:\srp\classification\DM-count master\classification_models/model2.pt')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def transform_image(image_bytes):
        my_transforms = transforms.Compose([transforms.Resize(1024),
                                            transforms.CenterCrop(384),
                                            transforms.ToTensor(),
                                            transforms.Normalize(
                                                [0.485, 0.456, 0.406],
                                                [0.229, 0.224, 0.225])])
        image = Image.open(io.BytesIO(image_bytes))
        image = image.convert('RGB')
        return my_transforms(image).unsqueeze(0)

    def get_prediction(image_bytes):
        tensor = transform_image(image_bytes=image_bytes)
        tensor=tensor.to(device)
        output1 = class_model1.forward(tensor)
        output2 = class_model2.forward(tensor)
        probs1 = torch.nn.functional.softmax(output1, dim=1)
        probs2 = torch.nn.functional.softmax(output2, dim=1)
        return probs1, probs2
        #conf, classes = torch.max(probs, 1)
        #return conf.item(), classes.item()
    #device = torch.device('cuda')
    
    #model_path = args.model_path
    crop_size = args.crop_size
    #data_path = args.data_path
    #model_p = 'D:\srp\classification\DM-count master\ckpts/JHU_high_input-384_wot-0.1_wtv-0.01_reg-10.0_nIter-100_normCood-0/best_model_7.pth'
    #data_p = 'D:\srp\classification\DM-count master\data\original\Shanghaitech/part_A/validation'
    #print(data_path)
    data_p = 'D:\srp\classification\DM-count master\data\original\Shanghaitech/part_A/validation' #'./data/original/shanghaitech/part_A_final/test_data/images/'
   # gt_path = 'D:\srp\classification\DM-count master\data\original\Shanghaitech/part_A/test_data'#'./data/original/shanghaitech/part_A_final/test_data/ground_truth_csv/'
    model_path_low = 'D:\srp\classification\DM-count master\dm_count_low/saved_models/low.pth'#'../crowdcount-mcnn-master-low/saved_models/mcnn_shtechA_164.h5'                #low

    output_dir = './output/'
    model_name = os.path.basename(model_path_low).split('.')[0]
    file_results = os.path.join(output_dir,'results_' + model_name + '_.txt')
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    output_dir = os.path.join(output_dir, 'density_maps_' + model_name)
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
        
    if args.dataset.lower() == 'qnrf':
        dataset = crowd.Crowd_qnrf(os.path.join(data_p, 'test'), crop_size, 8, method='val')
    elif args.dataset.lower() == 'nwpu':
        dataset = crowd.Crowd_nwpu(os.path.join(data_p, 'val'), crop_size, 8, method='val')
    elif args.dataset.lower() == 'sha' or args.dataset.lower() == 'shb':
        #dataset = crowd.Crowd_sh(os.path.join(data_path, 'test_data'), crop_size, 8, method='val')
        dataset = crowd.Crowd_sh((data_p), crop_size, 8, method='val')
    else:
        raise NotImplementedError
    dataloader = torch.utils.data.DataLoader(dataset, 1, shuffle=False,
                                             num_workers=1, pin_memory=True)
    
    
    #print(os.path.join(data_path, 'test_data'))
# =============================================================================
#     if args.pred_density_map_path:
#         #import cv2
#         if not os.path.exists(args.pred_density_map_path):
#             os.makedirs(args.pred_density_map_path)
# =============================================================================
    
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
    
    mae = 0.0
    mse = 0.0
    
    im_p = 'D:\srp\classification\DM-count master\data\original\Shanghaitech/part_A/validation/images'
    data_files = [filename for filename in os.listdir(im_p) \
                       if os.path.isfile(os.path.join(im_p,filename))]
    num_samples = len(data_files)
    for inputs, count, name in dataloader:
        image_path = os.path.join(im_p, str(name[0]) + '.jpg')  # for shutil
        inputs = inputs.to(device)
        #print(name,count,inputs)
        assert inputs.size(0) == 1, 'the batch size should equal to 1'
        #------------------------------------low
        image_bytes = open(image_path, 'rb').read()
        conf1, conf2 = get_prediction(image_bytes=image_bytes)
        with torch.set_grad_enabled(False):
            outputs, _ = net_low(inputs)
            et_count_low = torch.sum(outputs).item()
            
            outputs, _ = net_med(inputs)
            et_count_med = torch.sum(outputs).item()
            
            outputs, _ = net_high(inputs)
            et_count_high = torch.sum(outputs).item()
            
            gt_count = count[0].item()
            #et_count = np.sum(density_map)
            et_count = int(0.0 * (conf1[0][0] * et_count_high + conf1[0][1] * et_count_low + conf1[0][2] * et_count_med) + 1.0 * (conf2[0][0] * et_count_high + conf2[0][1] * et_count_low + conf2[0][2] * et_count_med))
            mae += abs(gt_count-et_count)
            mse += ((gt_count-et_count)*(gt_count-et_count))
            #if vis:
                #utils.display_results(inputs, gt_count, outputs)
            #if save_output:
                #utils.save_density_map(outputs, output_dir, 'output_' + str(name[0]) + '.jpg')
                
    mae = mae/num_samples
    mse = np.sqrt(mse/num_samples)
    print('\nMAE: %0.2f, MSE: %0.2f' % (mae,mse))

    f = open(file_results, 'w') 
    f.write('MAE: %0.2f, MSE: %0.2f' % (mae,mse))
    f.close()
        #img_err = count[0].item() - torch.sum(outputs).item()
    
        #print(name, img_err, count[0].item(), torch.sum(outputs).item())
        #image_errs.append(img_err)



#image_errs = np.array(image_errs)
#mse = np.sqrt(np.mean(np.square(image_errs)))
#mae = np.mean(np.abs(image_errs))
#print('{}: mae {}, mse {}\n'.format(model_path, mae, mse))


















