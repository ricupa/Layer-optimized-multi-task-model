import os
import copy
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
from torchvision import transforms, utils, models
import torchvision.transforms as transforms
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import pickle
import yaml
import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd 
from create_dataset import *
from torch.utils.data import Dataset, DataLoader
import wandb
from proxssi.penalties import *
from proxssi.groups.resnet_gp import resnet_groups
from proxssi.groups.vit_gp import vit_groups
from proxssi.optimizers.adamw_hf import AdamW
from sklearn.metrics import f1_score
# from proxssi.tests import penalties
from proxssi import penalties
import segmentation_models_pytorch as smp
from proxssi.optimizers.sp_term import *


def calculate_compression_ratio(total_params, non_zero_params):
    """
    Calculate the compression ratio.

    :param total_params: Total number of parameters.
    :param non_zero_params: Number of non-zero parameters.
    :return: Compression ratio.
    """
    if non_zero_params == 0:
        raise ValueError("Number of non-zero parameters cannot be zero.")
    
    return total_params / non_zero_params



def get_backbone(config,num_input_ch):
    """ Return the backbone """   
    
    num_input_ch = config['input_img_channels']    
    
    if config['backbone'] == 'resnetd50':
        from models.resnetd import ResNetD
        backbone = ResNetD('50', num_input_ch)
    elif config['backbone'] == 'resnetd101':
        from models.resnetd import ResNetD
        backbone = ResNetD('101', num_input_ch)        
    elif config['backbone'] == 'vit':
        backbone = ViTBackbone()
    else:
        print('backbone does not exist')
    
    bb_channels = backbone.channels    
    
    for ct, child in enumerate(backbone.children()): 
        for param in child.parameters():
            param.requires_grad = True   
        
    return backbone, bb_channels


def get_head(config, in_channels, task):
    from models.ASPP import DeepLabHead
    from models.class_head import ClassificationHead

    if config['dataset_name'] == 'NYU':
        config['NYU_num_classes']= {'segmentsemantic': config['seg_classes_NYU'], 'depth_euclidean': 1,'surface_normal': 3, 'edge_texture': 1 }   ### 41
        
        Head =  DeepLabHead(in_channels = in_channels,num_classes=config['NYU_num_classes'][task])

        for params in Head.parameters():
            params.requires_grad = True 
        
        ##### initialize the gradients to zero         
        # for param in Head.parameters():
        #     param.grad = torch.zeros_like(param)
    elif config['dataset_name'] == 'celebA':
        if task == 'segmentsemantic':
            Head = DeepLabHead(in_channels = in_channels,num_classes=3)      ######0/255:'background',1:'hair',2:'skin'
        else:
            Head = ClassificationHead(in_channels = in_channels, num_classes = 1)     ### binary classifications
        
        for params in Head.parameters():
            params.requires_grad = True 
        
        
    else:
        raise NotImplementedError('Task head for the dataset not found')

    return Head    
       


def get_model(config):
    from models.all_models import SingleTaskModel, MultiTaskModel
    
    backbone, bb_channels = get_backbone(config, config['num_input_ch'] )
    
    if config['setup'] == 'singletask':
        task = config['task_list'][0]
        print('initialize single-task model')   
        head = get_head(config, bb_channels[-1], task)
        model = SingleTaskModel(backbone, head, task)
        
    elif config['setup'] == 'multitask':   
        print('initialize multi-task model')     
        heads = torch.nn.ModuleDict({task: get_head(config, bb_channels[-1], task) for task in config['task_list']})      
        model = MultiTaskModel(backbone, heads, config['task_list'])
           
    else:
        raise NotImplementedError('Unknown setup {}, model not found'.format(config['setup']))   

    return model 



def get_loss(config, task):
    """ Return loss function for a specific task """
    
    if task == 'segmentsemantic':
        from loss_functions import Seg_loss
        criterion = Seg_loss(config)  


    elif task == 'depth_euclidean':
        from loss_functions import SSIMLoss, Depth_combined_loss, DepthMSELoss, RMSE_log
        criterion = Depth_combined_loss()
        # criterion = DepthMSELoss()
    
    elif task == 'surface_normal':
        from loss_functions import surface_normal_loss
        criterion = surface_normal_loss()
    
    elif task == 'edge_texture':
        from loss_functions import edge_loss
        criterion =  edge_loss()

        
        ###'High_Cheekbones', 'Big_Lips','Wearing_Lipstick'
    elif (task == 'class_male') or (task == 'class_eyebrows') or (task == 'class_glasses')or (task == 'class_smile') or (task == 'class_highcheekbones')or (task == 'class_biglips') or (task == 'class_lipstick'):
        criterion = nn.BCEWithLogitsLoss()
        
    else:
        raise NotImplementedError('Undefined Loss: Choose a task among '
                                    'class_object, class_scene, segmentsemantic')

    return criterion

def get_criterion(config):
    """ Return training criterion for a given setup """
    if config['setup'] == 'singletask':
        task = config['task_list'][0]
        loss_ft = get_loss(config, task)        

    elif config['setup'] == 'multitask':        
        loss_ft = torch.nn.ModuleDict({task: get_loss(config, task) for task in config['task_list']})                
               
    else:
        raise NotImplementedError('Loss function not found for setup {}'.format(config['setup']))    
    return loss_ft


def get_BB_optimizer(config, model):
    """ Return optimizer for a given model and setup """
    # l = torch.nn.Parameter(torch.tensor([0.01]), requires_grad=True)
    if (config['group_sparsity'] == True) and (config['bb_optimizer_params']['penalty'] == 'l1_l2'):
        # args = config['optimizer_params']
        args = {'weight_decay': config['bb_optimizer_params']['weight_decay'], 
                'learning_rate': config['bb_optimizer_params']['learning_rate']}
        if config['backbone'] == 'vit':
            grouped_params = vit_groups(model,args)
        else:            
            grouped_params = resnet_groups(model, args)
        prox_kwargs = {'lr': config['bb_optimizer_params']['learning_rate'], 
                        'lambda_': config['bb_optimizer_params']['lambda']}
        # optimizer_kwargs = {'lr': args.learning_rate,'penalty': penalty,'prox_kwargs': prox_kwargs}
        optimizer = AdamW(grouped_params, 
                          lr = config['bb_optimizer_params']['learning_rate'], 
                          betas = config['bb_optimizer_params']['betas'], 
                          weight_decay = config['bb_optimizer_params']['weight_decay'],
                          penalty = config['bb_optimizer_params']['penalty'],
                          prox_kwargs = prox_kwargs)
        
    else:           
        params = model.parameters()
        if config['bb_optimizer'] == 'adam':
            optimizer = torch.optim.Adam(params,lr=config['bb_optimizer_params']['learning_rate'],
                                         betas=config['bb_optimizer_params']['betas'], 
                                         weight_decay=config['bb_optimizer_params']['weight_decay'])

        elif config['bb_optimizer'] == 'sgd':
            optimizer = torch.optim.SGD(params,lr=config['bb_optimizer_params']['learning_rate'], 
                                        momentum=0.9, 
                                        weight_decay=config['bb_optimizer_params']['weight_decay'],
                                        nesterov = True)

        elif config['bb_optimizer'] == 'adamw':
            optimizer = torch.optim.AdamW(params, lr=config['bb_optimizer_params']['learning_rate'], 
                                          betas=config['bb_optimizer_params']['betas'], 
                                          weight_decay=config['bb_optimizer_params']['weight_decay'], 
                                          eps=1e-07, amsgrad= True)

        else:
            raise ValueError('Invalid optimizer {}'.format(config['optimizer']))   
    
    return optimizer



def get_task_optimizer(config, model):
    
    """ Return optimizer for a given model and setup """

    param_list = model.parameters()
    # if log_vars is not None:
    #     param_list = list(model.parameters()) + [log_vars]
    # else:
    #     param_list = model.parameters()
    
    # params_to_add = [nn.Parameter(param.data, requires_grad=True).to(config['device']) for param in log_vars.values()]
        
    # # Next, if log_vars is provided, add them to the list
    # if log_vars is not None:
    #     param_list = list(model.parameters()) + params_to_add
    # else:
        
    #     param_list = list(model.parameters())
        
    if config['optimizer'] == 'adam':
        optimizer = torch.optim.Adam(param_list ,lr=config['optimizer_params']['learning_rate'],
                                        betas=config['optimizer_params']['betas'], 
                                        weight_decay=config['optimizer_params']['weight_decay'])

    elif config['optimizer'] == 'sgd':
        optimizer = torch.optim.SGD(param_list ,lr=config['optimizer_params']['learning_rate'], 
                                    momentum=0.9, 
                                    weight_decay=config['optimizer_params']['weight_decay'],
                                    nesterov = True,
                                    foreach = True)

    elif config['optimizer'] == 'adamw':
        optimizer = torch.optim.AdamW(param_list , lr=config['optimizer_params']['learning_rate'], 
                                        betas=config['optimizer_params']['betas'], 
                                        weight_decay=config['optimizer_params']['weight_decay'], 
                                        eps=1e-07, amsgrad= True)

    else:
        raise ValueError('Invalid optimizer {}'.format(config['optimizer']))   

    return optimizer




def create_config(config_exp):
    with open(config_exp, 'r') as stream:
        # config = yaml.safe_load(stream)
        config = yaml.load(stream,Loader=yaml.Loader)

    return config  #### its a dictionary of all the parameters 



def get_transformations(config):
    ### return transformations 
    import utils_transforms as tr
    #tr.RandomRotate(30),
    img_size = config['input_shape']
    
    if config['backbone'] == 'vit':
        img_size= 224
        

    if config['dataset_name'] == 'NYU':
    #     transform_tr = [tr.ToTensor(), tr.FixedResize((img_size,img_size)), tr.RandomHorizontalFlip(), tr.RandomRotate(90)]
        
        transform_tr = [tr.ToTensor(), tr.FixedResize((img_size,img_size))]
        transform_v = [tr.ToTensor(), tr.FixedResize((img_size,img_size))]
        transform_t = [tr.ToTensor(), tr.FixedResize((img_size,img_size))]

        train_transform = transforms.Compose(transform_tr)
        val_transform = transforms.Compose(transform_v)       
        test_transform = transforms.Compose(transform_t)                 

    elif config['dataset_name'] == 'celebA':
        train_transform = transforms.Compose([transforms.ToTensor(),transforms.Resize((img_size, img_size))])
        val_transform = transforms.Compose([transforms.ToTensor(),transforms.Resize((img_size, img_size))])
        test_transform = transforms.Compose([transforms.ToTensor(),transforms.Resize((img_size, img_size))])
        
    else:
        print('invalid dataset name')
        
    return train_transform, val_transform, test_transform



def get_dataset(config):
    
    if config['dataset_name'] == 'NYU':        

        train_transform, val_transform, test_transform = get_transformations(config)

        train_dataset = NYUDataset(config = config, data_dir = config['data_dir_NYU'], set = 'train', transform= train_transform)

        val_dataset = NYUDataset(config = config, data_dir = config['data_dir_NYU'], set = 'val', transform= val_transform)
        
        test_dataset = NYUDataset(config = config, data_dir = config['data_dir_NYU'], set = 'test', transform= test_transform)
        
    elif config['dataset_name'] == 'celebA':
        
        train_transform, val_transform, test_transform = get_transformations(config)
        
        train_dataset = CelebMaskHQDataset(config = config, set = 'train', transform=train_transform)
        
        val_dataset = CelebMaskHQDataset(config = config, set = 'val', transform=val_transform)
        
        test_dataset = CelebMaskHQDataset(config = config, set = 'test', transform=test_transform)
        
        # train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset,[0.6,0.2,0.2])



    else:
        print(" invalid dataset name")
    return train_dataset, val_dataset, test_dataset



def collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))   
    return torch.utils.data.dataloader.default_collate(batch)



def get_dataloader(config,train_dataset,val_dataset,test_dataset):   

    train_loader = DataLoader(dataset = train_dataset, batch_size = config['train_batch_size'], shuffle = True, num_workers = config['num_workers'], collate_fn=collate_fn, drop_last=True)   

    if 'meta' in config.keys():
        val_loader = DataLoader(dataset = val_dataset, batch_size = config['val_batch_size'], shuffle = True, num_workers = config['num_workers'],collate_fn=collate_fn, drop_last=True)
    else:
        val_loader = DataLoader(dataset = val_dataset, batch_size = config['val_batch_size'], shuffle = False, num_workers = config['num_workers'],collate_fn=collate_fn, drop_last=True)
        
    test_loader = DataLoader(dataset = test_dataset, batch_size = config['test_batch_size'], shuffle = False, num_workers = config['num_workers'],collate_fn=collate_fn, drop_last=True)

    return train_loader, val_loader, test_loader

def compute_valid_depth_mask(d1, d2=None):
    """Computes the mask of valid values for one or two depth maps    
    Returns a valid mask that only selects values that are valid depth value 
    in both depth maps (if d2 is given).
    Valid depth values are >0 and finite.
    """

    if d2 is None:
        valid_mask = torch.isfinite(d1)
        valid_mask[valid_mask] = (d1[valid_mask] > 0)
    else:
        valid_mask = torch.isfinite(d1) & torch.isfinite(d2)
        _valid_mask = valid_mask.clone()
        valid_mask[_valid_mask] = (d1[_valid_mask] > 0) & (d2[_valid_mask] > 0)
    return valid_mask



def get_valid_depth_values(pred,gt):
    valid_mask = torch.tensor(compute_valid_depth_mask(pred, gt),dtype=torch.uint8)
    depth_pred = torch.nan_to_num(pred*valid_mask, nan=0.0, posinf=1.0, neginf=0.0 )
    depth_gt = torch.nan_to_num(gt*valid_mask)
    return depth_pred, depth_gt

class Sobel(nn.Module):
    def __init__(self):
        super(Sobel, self).__init__()
        self.edge_conv = nn.Conv2d(1, 2, kernel_size=3, stride=1, padding=1, bias=False)
        edge_kx = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
        edge_ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
        edge_k = np.stack((edge_kx, edge_ky))

        edge_k = torch.from_numpy(edge_k).float().view(2, 1, 3, 3)
        self.edge_conv.weight = nn.Parameter(edge_k)
        
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        out = self.edge_conv(x) 
        out = out.contiguous().view(-1, 2, x.size(2), x.size(3))
  
        return out
    
    
class ModelWrapper(nn.Module):
    def __init__(self, config, loss_ft: nn.ModuleDict):
        super(ModelWrapper, self).__init__()
        self.set_up = config['setup']
        self.loss_ft = loss_ft
        self.tasks = config['task_list']
        self.task = self.tasks[0]  # for single task only
        self.comb_loss = config['comb_loss']  # mode for balancing the loss
        self.device = config['device']
        self.flags = {task: 1 for task in self.tasks}

        if self.comb_loss == 'wt_sum':  # (self.comb_loss == 'uncertainity') or
            self.log_vars = nn.Parameter(torch.ones(len(self.tasks), device=self.device, requires_grad=True))
        elif self.comb_loss == 'sum':
            self.weights = {task: torch.tensor(1.0) for task in self.tasks}
        

        self.config = config
        

    def _apply_activations(self, preds):
        for task, pred in preds.items():
            if task in ['depth_euclidean', 'edge_texture']:
                preds[task] = torch.nn.Sigmoid()(pred)
            elif 'surface_normal' in task:
                preds[task] = torch.nn.Tanh()(pred)
        return preds

    def _handle_depth_euclidean(self, pred, targets):
        new_dim = pred['depth_euclidean'].shape[-2:]
        target_depth = F.interpolate(targets['depth_euclidean'].type(torch.DoubleTensor), size=new_dim, mode="bilinear")
        pred['depth_euclidean'], targets['depth_euclidean'] = get_valid_depth_values(pred['depth_euclidean'], target_depth.to(self.device))
        assert targets['depth_euclidean'].shape == pred['depth_euclidean'].shape

    def single_task_forward(self, model, images, targets, phase):
        pred = model(images)
        pred = self._apply_activations(pred)

        if 'depth_euclidean' in self.tasks:
            self._handle_depth_euclidean(pred, targets)

        out= {}
        out = {self.tasks[0]: self.loss_ft(pred[self.tasks[0]], targets[self.tasks[0]].float())}
        # out = {task: self.loss_ft[task](pred[task], targets[task].float()) for task in self.tasks}

        if self.config.get('group_sparsity') and phase == 'train':
            if self.config['bb_optimizer_params']['penalty']== 'l1_l2':
                sp_term = GroupSparsityTerm(self.config, model)
                out[self.task] += sp_term()
            elif self.config['bb_optimizer_params']['penalty']== 'l1':
                reg_fn = L1Sparsity(self.config)
                out[self.task] += reg_fn.compute_loss(model)
            else:
                raise KeyError('invalid penalty type')
            
            # sp_term = GroupSparsityTerm(self.config, model)
            # reg_term = sp_term()
            # out = {'total': out[self.task] + reg_term}
        # else:
        #     out['total'] =  out[self.task].requires_grad_()
        
        out = {'total': out[self.task].requires_grad_()}  ## for single task the total and task specific loss is the same
        return pred, targets, out

    def multi_task_forward(self, model, images, targets, phase):
        pred = model(images)
        pred = self._apply_activations(pred)

        if 'depth_euclidean' in self.tasks:
            self._handle_depth_euclidean(pred, targets)
            
        out = {}
        out = {task: self.loss_ft[task](pred[task], targets[task].float()) for task in self.tasks}

        if self.comb_loss == 'sum':
            weighted_losses = {task: self.weights[task] * loss for task, loss in out.items()}
            total_loss = sum(weighted_losses[task] * self.flags[task] for task in self.tasks)
        elif self.comb_loss == 'uncertainity':
            weighted_losses = {task: torch.exp(-getattr(model, task)) * out[task] + getattr(model, task) for task in self.tasks}
            total_loss = sum(weighted_losses[task] * self.flags[task] for task in self.tasks)
        else:
            raise ValueError('Implementation error: this mode of combining loss is not implemented.')

        if self.config.get('group_sparsity') and phase == 'train':
            if self.config['bb_optimizer_params']['penalty']== 'l1_l2':
                sp_term = GroupSparsityTerm(self.config, model)
                total_loss += sp_term()
            elif self.config['bb_optimizer_params']['penalty']== 'l1':
                reg_fn = L1Sparsity(self.config)
                total_loss += reg_fn.compute_loss(model)
            else:
                raise KeyError('invalid penalty type') 
            
        out['total'] =  total_loss.requires_grad_()

        return pred, targets, out
    
    
def calculate_percentage_sparsity_params(model):
    num_layer = 0
    num_nonzero = 0
    num_zero = 0
    total_params = 0
    non_zero_params = 0
    for name, param in model.named_parameters():
        if ('backbone' in name) and ('weight' in name) and (len(param.shape) == 4):
            
            for i in range(param.shape[1]):
                num_layer += 1
                total_params += len(param[:,i,:,:].flatten())
                
                if torch.nonzero(param[:,i,:,:]).size(0) > 0:
                    num_nonzero += 1  ### for group sparsity onlly 
                    # non_zero_params += len(param[i,:,:,:].flatten())
                    non_zero_params += torch.nonzero(param[:,i,:,:]).size(0)
                else:
                    num_zero += 1
                    
    assert num_layer == (num_nonzero + num_zero)    
    spar = (num_zero / num_layer)*100    ### this is group sparsity
    
    sparsity_params = ((total_params - non_zero_params)/total_params)*100
    zero_params = total_params - non_zero_params  
    return spar, sparsity_params, total_params, zero_params




def evaluation_metrics(task_list,config,output,label):    ###label is gt
    metrics = {}
    for task in task_list:
        
        if (task == 'class_male') or (task == 'class_eyebrows') or (task == 'class_glasses')or (task == 'class_smile')or (task == 'class_highcheekbones')or (task == 'class_biglips') or (task == 'class_lipstick'):
            m={}
            
            gt = label[task].detach().cpu().numpy()
            pred = output[task].detach().cpu().numpy()
            
            # gt = label[task]
            # pred = output[task]
            
            assert(gt.shape == pred.shape)
            threshold = 0.3
            
            out = (pred >= threshold)*1.0
            correct = (out == gt)*1.0
        
            # calculate the accuracy
            m['accuracy'] = correct.sum() / len(correct) 
            metrics[task] = m



        elif task == 'segmentsemantic':
            from seg_metrics import Seg_cross_entropy_metric, calculate_IoU
            
            m = {}   
            pred = output[task]
            num_classes = pred.shape[1]
            new_dim = pred.shape[-2:] 
            gt = label[task] 
            
            gt = F.interpolate(gt, size=new_dim, mode='nearest')              
            gt = torch.squeeze(gt,1) 
            gt = gt.type(torch.int64)   

            pred = nn.Softmax(dim=1)(pred)
            pred = torch.argmax(pred, dim = 1)
            assert pred.shape == gt.shape
            
            tp, fp, fn, tn = smp.metrics.functional.get_stats(pred, gt, mode = 'multiclass', ignore_index= 255, num_classes=num_classes)  
            m['IoU'] = smp.metrics.functional.iou_score(tp, fp, fn, tn, reduction='micro-imagewise', zero_division='warn')
            
            # IOU = calculate_IoU()  
            # # CE = Seg_cross_entropy_metric(config)             
            # # m['CrossEntropy'] = CE(pred, gt)    # ingore index 0 i.e. bg 
            # m['IoU'] = IOU(gt, pred)
            metrics[task] = m  
             

        elif task =='depth_euclidean':
            from depth_metrics import get_depth_metric     
            m = {}  
            new_dim= output[task].shape[-2:]
            label[task] = F.interpolate(label[task], size=new_dim)          

            if config['dataset_name'] == 'NYU':
                # binary_mask = (torch.sum(label[task], dim=1) > 3 * 1e-5).unsqueeze(1).to(config['device']) 
                binary_mask = (label[task] != 0).to(config['device']) 

            # elif config['dataset_name'] == 'Taskonomy':
            #     # label= label[task]*255
            #     # binary_mask = (label[task] != 1).to(config['device']) 
            #     binary_mask = (label[task] != 0).to(config['device']) 

            else:
                print('task not found')
            
            new_dim= output[task].shape[-2:]
            label[task] = F.interpolate(label[task], size=new_dim) 
            if config['dataset_name'] in ['NYU', 'Taskonomy']:
                m['mae'], _, _ , _ , m['rmse'], _ = get_depth_metric(output[task].to(config['device']) , label[task].to(config['device']) , binary_mask)
                metrics[task] = m
            else:
                print('task not found')

            

        elif task =='surface_normal':
            from metrics import sn_metrics
            m = {}
            
            gt = label[task]
            new_shape = gt.shape[-2:]
            pred = F.interpolate(output[task], size= new_shape, mode="bilinear") 
            m['cos_similarity'], angles = sn_metrics(pred, gt)

            if config['dataset_name'] == 'NYU':
                m['Angle Mean'] = np.mean(angles)
                m['Angle Median'] = np.median(angles)
                m['Angle RMSE'] = np.sqrt(np.mean(angles ** 2))
                m['Angle 11.25'] = np.mean(np.less_equal(angles, 11.25)) * 100
                m['Angle 22.5'] = np.mean(np.less_equal(angles, 22.5)) * 100
                m['Angle 30'] = np.mean(np.less_equal(angles, 30.0)) * 100
                m['Angle 45'] = np.mean(np.less_equal(angles, 45.0)) * 100
            
            metrics[task] = m 

        elif task == 'edge_texture':
            from metrics import edge_metrics
            m = {}
            gt = label[task].detach().cpu()
            pred = output[task].detach().cpu()           
            new_shape = pred.shape[-2:]            
            gt = F.interpolate(gt.float(), size= new_shape, mode="bilinear") 
            m['abs_err'] = edge_metrics(pred, gt)
            metrics[task] = m 

        else:
            print('unknown task for metric calculation')
    
    return metrics






def get_class_labels(config):
    if config['dataset_name'] == 'NYU':
        class_labels = {0: 'otherprop',
                        1: 'wall',
                        2: 'floor',
                        3: 'cabinet',
                        4: 'bed',
                        5: 'chair',
                        6: 'sofa',
                        7: 'table',
                        8: 'door',
                        9: 'window',
                        10: 'bookshelf',
                        11: 'picture',
                        12: 'counter',
                        13: 'blinds',
                        14: 'desk',
                        15: 'shelves',
                        16: 'curtain',
                        17: 'dresser',
                        18: 'pillow',
                        19: 'mirror',
                        20: 'floor mat',
                        21: 'clothes',
                        22: 'ceiling',
                        23: 'books',
                        24: 'refridgerator',
                        25: 'television',
                        26: 'paper',
                        27: 'towel',
                        28: 'shower curtain',
                        29: 'box',
                        30: 'whiteboard',
                        31: 'person',
                        32: 'night stand',
                        33: 'toilet',
                        34: 'sink',
                        35: 'lamp',
                        36: 'bathtub',
                        37: 'bag',    
                        38: 'otherstructure',   ##### categorized as other class 0
                        39: 'otherfurniture'}   ##### categorized as other class 0
        
      
        
    else:
        print('No class labels for given dataset')
    return class_labels



def draw_segmentation_map_NYU(outputs):
    
    label_map = [               
            #    (0,0,0), # background
            #    (255,255,255),  # wall
               (0, 255, 0),
               (0, 0, 255),
               (128, 0, 0), # 
               (0, 128, 0), # 
               (128, 128, 0), # 
               (0, 0, 128), # 
               (128, 0, 128), # 
               (0, 128, 128), # 
               (128, 128, 128), # 
               (64, 0, 0), # 
               (192, 0, 0), # 
               (64, 128, 0), # 
               (192, 128, 0), # 
               (64, 0, 128), # 
               (192, 0, 128), # 
               (64, 128, 128), # 
               (192, 128, 128), #
               (0, 64, 0), # 
               (0, 0, 64),
               (0, 192, 0),
               (0, 0, 192),
               (32,0,0),
               (0, 32, 0),
               (0, 0, 32),
               (255,128,128),
               (64,64, 128),
               (32, 255, 0),
               (255, 32, 0),
               (0, 32, 255),
               (192, 0,32),
               (32, 0, 192),
               (64, 32, 128),
               (128, 128, 32),
               (128, 64, 32),
               (192, 32, 64),
               (0, 192, 64),
               (192, 0, 255),
               (255, 64, 192),
               (0,0,0),
               (255,255,255) #### background, wall               
            ]
    labels = outputs.detach().cpu().numpy()    
    # labels = torch.argmax(outputs, dim=0).detach().cpu().numpy() 
    red_map = np.zeros_like(labels).astype(np.uint8)
    green_map = np.zeros_like(labels).astype(np.uint8)
    blue_map = np.zeros_like(labels).astype(np.uint8)
    
    for label_num in range(0, len(label_map)):
        index = labels == label_num
        red_map[index] = np.array(label_map)[label_num, 0]
        green_map[index] = np.array(label_map)[label_num, 1]
        blue_map[index] = np.array(label_map)[label_num, 2]
        
    segmented_image = np.stack([red_map, green_map, blue_map], axis=2)
    return torch.tensor(segmented_image)