import os
import cv2
import torch
import numpy as np
import time
import joblib
from config import Config
from PIL import Image
from matplotlib import pyplot as plt
from torch.nn import DataParallel
import torch.nn.functional as F
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, SubsetRandomSampler
from facenet_pytorch import MTCNN, InceptionResnetV1
import argparse
from attack.utils import load_ground_truth, Normalize, gkern, DI, get_gaussian_kernel
from models import *
from attack import stick
from mtcnn_pytorch_master.test import crop_face


trans = transforms.Compose([
                transforms.ToTensor(),
                #transforms.Normalize(mean=[0.5, 0.5, 0.5],std=[0.5, 0.5, 0.5])
            ])
#localtime = time.asctime( time.localtime(time.time()) )
inputsize = {'arcface50':[112,112],'cosface50':[112,112],'arcface34':[112,112],'cosface34':[112,112],
             'facenet':[160,160],'insightface':[112,112],
             'sphere20a':[112,96],'re_sphere20a':[112,96],'mobilefacenet':[112,112]}

def reward_slope(adv_face_ts, params_slove, sticker,device):
    advface_ts = adv_face_ts.to(device)
    x, y = params_slove[0]
    w, h = sticker.size
    advstk_ts = advface_ts[:,:,y:y+h,x:x+w]
    advstk_ts.data = advstk_ts.data.clamp(1/255.,224/255.)
    w = torch.arctanh(2*advstk_ts-1)
    x_wv = 1/2 - (torch.tanh(w)**2)/2
    mean_slope = torch.mean(x_wv)
    #print(w,x_wv)
    return mean_slope

def load_model(model_name, device):
    if(model_name == 'facenet'):
        resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
        return resnet
    elif (model_name == 'insightface'):
        insightface_path = '/home/guoying/rlpatch/stmodels/insightface/insightface.pth'
        model = Backbone(50,0.6,'ir_se')
        model.load_state_dict(torch.load(eval("{}_path".format(model_name)),map_location=device))
        model.eval()
        model = model.to(device)
        return model
    elif (model_name == 'sphere20a'):
        sphere20a_path = '/home/guoying/rlpatch/stmodels/sphere20a/sphere20a.pth'
        model = sphere20a(feature=True)
        model.load_state_dict(torch.load(eval("{}_path".format(model_name)),map_location=device))
        model.eval()
        model = model.to(device)
        return model
    elif (model_name == 're_sphere20a'):
        sphere20a_path = '/home/guoying/rlpatch/stmodels/re_sphere20a/re_sphere20a.pth'
        model = sphere20a(feature=True)
        #model.load_state_dict(torch.load(eval("{}_path".format(model_name)),map_location=device))
        model.eval()
        model = model.to(device)
        return model
    elif (model_name == 'mobilefacenet'):
        mobilefacenet_path = '/home/guoying/rlpatch/stmodels/mobilefacenet/mobilefacenet_scripted.pt'
        #model = MobileFaceNet()
        #model.load_state_dict(torch.load(eval("{}_path".format(model_name)),map_location=device))
        model = torch.jit.load(eval("{}_path".format(model_name)),map_location=device)
        model.eval()
        model = model.to(device)
        return model
    elif (model_name == 'arcface34'):
        arcface34_path = '/home/guoying/rlpatch/stmodels/arcface34/arcface_34.pth'
        model = iresnet34(False, dropout=0, fp16=True)
        model.load_state_dict(torch.load(eval("{}_path".format(model_name)),map_location=device))
        model.eval()
        model = model.to(device)
        return model
    elif (model_name == 'cosface34'):
        cosface34_path = '/home/guoying/rlpatch/stmodels/cosface34/cosface_34.pth'
        model = iresnet34(False, dropout=0, fp16=True)
        model.load_state_dict(torch.load(eval("{}_path".format(model_name)),map_location=device))
        model.eval()
        model = model.to(device)
        return model
    else:
        arcface50_path = '/home/guoying/rlpatch/stmodels/arcface50/ms1mv3_arcface_r50_fp16.pth'
        cosface50_path = '/home/guoying/rlpatch/stmodels/cosface50/glint360k_cosface_r50_fp16_0.1.pth'
        model = iresnet50(False, dropout=0, fp16=True)
        model.load_state_dict(torch.load(eval("{}_path".format(model_name)),map_location=device))
        model.eval()
        model = model.to(device)
        return model

def load_anchors(model_name, device, target):
    anchor_embeddings =  joblib.load('/home/guoying/rlpatch/stmodels/{}/embeddings_{}_5751.pkl'.format(model_name,model_name))
    anchor = anchor_embeddings[target:target+1]
    anchor = anchor.to(device)
    return anchor

def make_stmask(face,sticker,x,y):
    w,h = face.size
    mask = stick.make_masktensor(w,h,sticker,x,y)
    return mask

def crop_imgs(imgs,w,h):
    crops_result = []
    crops_tensor = []
    for i in range(len(imgs)):
        crop = crop_face(imgs[i],w,h)
        crop_ts = trans(crop)
        crops_result.append(crop)
        crops_tensor.append(crop_ts)
    return crops_result, crops_tensor

def cosin_metric(prd,src,device):
    nlen = len(prd)
    mlt = torch.zeros((nlen,1)).to(device)
    src_t = torch.t(src)
    #print(prd.shape,src_t.shape)
    for i in range(nlen):
        #print(prd[i].shape,src_t[:,i].shape)
        mlt[i] = torch.mm(torch.unsqueeze(prd[i],0),torch.unsqueeze(src_t[:,i],1))
    norm_x1 = torch.norm(prd,dim=1)
    norm_x1 = torch.unsqueeze(norm_x1,1)
    norm_x2 = torch.norm(src_t,dim=0)
    norm_x2 = torch.unsqueeze(norm_x2,1)
    #print('norm_x1,norm_x2 ',norm_x1.shape,norm_x2.shape)
    denominator = torch.mul(norm_x1, norm_x2)
    metrics = torch.mul(mlt,1/denominator)
    return metrics

def tiattack_face(x, y, epsilon, weights, model_names,
                  img, label, target, device, sticker,
                  width, height, emp_iterations, di, adv_img_folder, targeted = True):
    flag = -1 if targeted else 1
    liner_interval = [-4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0]
    # TI 参数设置
    channels=3                                             # 3通道
    kernel_size=5                                          # kernel大小
    kernel = gkern(kernel_size, 1).astype(np.float32)      # 3表述kernel内元素值得上下限
    gaussian_kernel = np.stack([kernel, kernel, kernel])   # 5*5*3
    gaussian_kernel = np.expand_dims(gaussian_kernel, 1)   # 1*5*5*3
    gaussian_kernel = torch.from_numpy(gaussian_kernel).to(device)  # tensor and cuda
    gaussian_filter = nn.Conv2d(in_channels=channels, out_channels=channels,
                                    kernel_size=kernel_size, groups=channels, bias=False, padding=7)
    gaussian_filter.weight.data = gaussian_kernel          # 高斯滤波，高斯核的赋值
    
    crops_result, crops_tensor = crop_imgs([img], width, height)
    X_ori = torch.stack(crops_tensor).to(device)
    #print(X_ori.shape)
    delta = torch.zeros_like(X_ori,requires_grad=True).to(device)
    #label = torch.tensor(label).to(device)

    fr_models, anchors = [], []
    for name in model_names:
        model = load_model(name, device)
        anchor = load_anchors(name, device, target)
        fr_models.append(model)
        anchors.append(anchor)
    #print(anchors.shape)
    mask = make_stmask(crops_result[0],sticker,x,y)

    for itr in range(emp_iterations):
        g_temp = []
        for t in range(len(liner_interval)):
            c = liner_interval[t]
            X_adv = X_ori + c * delta
            accm = 0
            for (i, name) in enumerate(model_names):
                X_op = nn.functional.interpolate(X_adv, (inputsize[name][0], inputsize[name][1]), mode='bilinear', align_corners=False)
                # if di:
                #     X_adv = X_ori + delta
                #     X_adv = DI(X_adv, 500)   # diverse input operation
                #     X_op = nn.functional.interpolate(X_adv, (inputsize[name][0], inputsize[name][1]), mode='bilinear', align_corners=False)
                feature = fr_models[i](X_op)
                l_sim = cosin_metric(feature,anchors[i],device)
                accm += l_sim * weights[i]
            #print('---iter {} interval {}--- loss = {}'.format(itr,t,loss))
            loss = flag * accm
            #print('---iter {} interval {}--- loss = {}'.format(itr,t,loss))
            loss.backward()
            
            # TI operation
            grad_c = delta.grad.clone()                        
            grad_c = F.conv2d(grad_c, gaussian_kernel, bias=None, stride=1, padding=(2,2), groups=3)
            #grad_a = grad_c / torch.mean(torch.abs(grad_c), (1, 2, 3), keepdim=True)+0.5*grad_momentum   # 1
            grad_a = grad_c
            # grad_momentum = grad_a
            g_temp.append(grad_a)
        g_syn = 0.0
        for j in range(9):
            g_syn += g_temp[j]
        g_syn = g_syn / 9.0
        delta.grad.zero_()
        # L-inf attack
        delta.data=delta.data-epsilon * torch.sign(g_syn)
        delta.data = delta.data * mask.to(device)
        #delta.data=delta.data.clamp(-args.linf_epsilon/255.,args.linf_epsilon/255.)
        delta.data=((X_ori+delta.data).clamp(0,1))-X_ori              # 噪声截取操作
        
        with torch.no_grad():
            X_adv = X_ori + delta
            accm = 0
            for (i, name) in enumerate(model_names):
                X_op = nn.functional.interpolate(X_adv, (inputsize[name][0], inputsize[name][1]), mode='bilinear', align_corners=False)
                feature = fr_models[i](X_op)
                l_sim = cosin_metric(feature,anchors[i],device).item()
                accm += l_sim * weights[i]
            print('---iter {} --- loss = {}'.format(itr,flag * accm))
    
    adv_final = (X_ori+delta)[0].cpu().detach().numpy()
    adv_final = (adv_final*255).astype(np.uint8)
    localtime1 = time.asctime( time.localtime(time.time()) )
    file_path = os.path.join(adv_img_folder, '{}.jpg'.format(localtime1))
    adv_x_255 = np.transpose(adv_final, (1, 2, 0))
    im = Image.fromarray(adv_x_255)
    im.save(file_path,quality=99)
    torch.cuda.empty_cache()

def miattack_face(params_slove, model_names,
                  img, label, target, device, sticker,
                  width, height, emp_iterations, di, adv_img_folder, targeted = True):
    x, y = params_slove[0]
    weights = params_slove[1]
    epsilon = params_slove[2]
    flag = -1 if targeted else 1
    w,h = img.size
    if(w!=width or h!=height):
        crops_result, crops_tensor = crop_imgs([img], width, height)
    else:
        crops_result = [img]
        crops_tensor = [trans(img)]
    X_ori = torch.stack(crops_tensor).to(device)
    #print(X_ori.shape)
    delta = torch.zeros_like(X_ori,requires_grad=True).to(device)
    #label = torch.tensor(label).to(device)
    
    fr_models, anchors = [], []
    for name in model_names:
        model = load_model(name, device)
        anchor = load_anchors(name, device, target)
        fr_models.append(model)
        anchors.append(anchor)
        
    mask = make_stmask(crops_result[0],sticker,x,y)
    grad_momentum = 0
    for itr in range(emp_iterations):   # iterations in the generation of adversarial examples
        X_adv = X_ori + delta
        accm = 0
        print('---iter {}---'.format(itr),end=' ')
        for (i, name) in enumerate(model_names):
            X_op = nn.functional.interpolate(X_adv, (inputsize[name][0], inputsize[name][1]), mode='bilinear', align_corners=False)
            feature = fr_models[i](X_op)
            l_sim = cosin_metric(feature,anchors[i],device)
            print(name,':','{:.4f}'.format(l_sim.item()),end=' ')
            accm += l_sim * weights[i]
        #print('---iter {} interval {}--- loss = {}'.format(itr,t,loss))
        slope = reward_slope(X_adv,params_slove,sticker,device)
        loss = flag * accm - 0.1*slope
        print('L_sim = {:.4f},L_slope = {:.4f}'.format(flag * accm.item(),slope.item()),end='\r')
        loss.backward()
        
        # MI operation
        grad_c = delta.grad.clone()                        
        grad_a = grad_c / torch.mean(torch.abs(grad_c), (1, 2, 3), keepdim=True)+1.0*grad_momentum   # 1
        grad_momentum = grad_a
            
        delta.grad.zero_()
        delta.data=delta.data-epsilon * torch.sign(grad_momentum)
        delta.data = delta.data * mask.to(device)
        #delta.data=delta.data.clamp(-args.linf_epsilon/255.,args.linf_epsilon/255.)
        delta.data=((X_ori+delta.data).clamp(0,1))-X_ori

        # with torch.no_grad():
        #     X_adv = X_ori + delta
        #     accm = 0
        #     print('---iter {}---'.format(itr),end=' ')
        #     for (i, name) in enumerate(model_names):
        #         X_op = nn.functional.interpolate(X_adv, (inputsize[name][0], inputsize[name][1]), mode='bilinear', align_corners=False)
        #         feature = fr_models[i](X_op)
        #         l_sim = cosin_metric(feature,anchors[i],device).item()
        #         print(name,':','{:.4f}'.format(l_sim),end=' ')
        #         accm += l_sim * weights[i]
        #     print('loss = {:.4f}'.format(flag * accm),end='\r')

    #joblib.dump(X_op.cpu().detach(), 'X_op.pkl')
    adv_face_ts = (X_ori+delta).cpu().detach()
    adv_final = (X_ori+delta)[0].cpu().detach().numpy()
    adv_final = (adv_final*255).astype(np.uint8)
    localtime2 = time.asctime( time.localtime(time.time()) )
    file_path = os.path.join(adv_img_folder, '{}.jpg'.format(localtime2))
    adv_x_255 = np.transpose(adv_final, (1, 2, 0))
    im = Image.fromarray(adv_x_255)
    #im.save(file_path,quality=99)
    #torch.cuda.empty_cache()
    return adv_face_ts,im,mask

# def check():
#     # faceimg = Image.open('/home/guoying/rlpatch/example/guoying/1103_gy.jpg')
#     # crops_result, crops_tensor = crop_imgs([faceimg],args.width, args.height)
#     crops_result = [Image.open('/home/guoying/rlpatch/adv_imgs/0.jpg')]
#     anchor_embeddings =  joblib.load('/home/guoying/rlpatch/stmodels/{}/embeddings_{}_5752.pkl'.format(args.source_model,args.source_model))
#     anchors = anchor_embeddings[5749:5750]
#     anchors = anchors.to(device)

#     intput = torch.unsqueeze(trans(crops_result[0]),0).to(device)
#     print(intput.shape)
#     intput = nn.functional.interpolate(intput, (inputsize[0], inputsize[1]), mode='bilinear', align_corners=False)            # 插值到224
#     feature = fr_model(intput)
#     m = cosin_metric(feature,anchors,device)
#     print(m)

if __name__=="__main__":
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    args = parse_arguments()
    dataset = datasets.ImageFolder(args.input_dir)
    dataset.idx_to_class = {i:c for c, i in dataset.class_to_idx.items()}
    
    def collate_fn(x):
        return x
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn
    )

    fr_model = load_model(args.source_model,device).eval().to(device)
    if(args.source_model == 'facenet'):
        inputsize = [160,160]
    else:
        inputsize = [112,112]
    #check()
    #miattack_face(loader,fr_model)
    
    #x,y = 65,45
