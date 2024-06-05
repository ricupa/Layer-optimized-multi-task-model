import os
import gc
import torch
gc.collect()
torch.cuda.empty_cache()
os.environ["CUDA_LAUNCH_BLOCKING"]= '1'
# os.enviorn['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:20000'
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]= "0"
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True
import numpy as np
import sys
# import torch
from termcolor import colored
import pandas as pd
import collections
from collections import defaultdict, abc
import time
import yaml
import argparse
# import dill as pickle
import warnings
warnings.filterwarnings('ignore')
# from torchsummary import summary
from pytorchtools import EarlyStopping
from utils import *
from tqdm import tqdm
import wandb
import json
from proxssi.groups.resnet_gp import resnet_groups
from proxssi.optimizers.adamw_hf import AdamW
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
# from proxssi.tests import penalties
from proxssi import penalties
from torchvision.utils import make_grid
from create_dataset_few_shot import *
parser = argparse.ArgumentParser(description='Training')
parser.add_argument('--config_exp', help='Config file for the experiment')
args = parser.parse_args()


NYU_resnetd50 = {'dir_path' : '/proj/ltu_mtl/users/x_ricup/MTL_adaptive_results/meta/NYU', 
                 'segmentsemantic' : {'name': 'Exp1_single_seg_sparse1e_3_trial_', 'trials': 5},
                 'depth_euclidean' : {'name': 'Exp2_single_depth_sparse1e_3_trial_', 'trials': 5},
                 'surface_normal' : {'name': 'Exp3_single_sn_sparse1e_3_trial_', 'trials': 5},
                 'edge_texture' : {'name': 'Exp4_single_edge_sparse1e_3_trial_', 'trials': 5}
                 }



def get_pretrained_weights(config, model):
    if config['backbone'] == 'resnetd50':
        task_info = NYU_resnetd50
        
    backbone_params = defaultdict(list) 
    decoders_params = {task: defaultdict(list) for task in config['task_list']}
    
    for task in config['task_list']:
        
        
        for trial in range(task_info[task]['trials']):
            # print(task, task_info[task]['name'], trial)
            
            model_path = os.path.join(task_info['dir_path'], f"{task_info[task]['name']}{trial}", 'checkpoint.pt')
            state_dict = torch.load(model_path, map_location= config['device'])  # Adjust map_location if necessary
            
            # Accumulate parameters
            for name, param in state_dict.items():
                if name.startswith('backbone.'):  # Assuming backbone parameters are prefixed with 'backbone.'
                    backbone_params[name].append(param)
                elif name.startswith(f'decoders.{task}.'):  # Assuming decoder parameters are prefixed with 'decoders.{task}.'
                    decoders_params[task][name].append(param)

    # Average backbone parameters
    average_backbone_params = {name: torch.mean(torch.stack(params_list), dim=0) for name, params_list in backbone_params.items()}
    
    # Load averaged backbone parameters into the model
    model_backbone_state_dict = {k.replace('backbone.', ''): v for k, v in model.backbone.state_dict().items()}
    filtered_backbone_params = {k: v for k, v in average_backbone_params.items() if k in model_backbone_state_dict}
    model.backbone.load_state_dict(filtered_backbone_params, strict=False)

    # Average and load decoder parameters for each task
    for task in config['task_list']:
        average_decoder_params = {name: torch.mean(torch.stack(params_list), dim=0) for name, params_list in decoders_params[task].items()}
        
        # Prepare decoder parameters for loading
        model_decoder_state_dict = {k.replace(f'decoders.{task}.', ''): v for k, v in model.decoders[task].state_dict().items()}
        filtered_decoder_params = {k: v for k, v in average_decoder_params.items() if k in model_decoder_state_dict}
        
        # Load averaged decoder parameters
        model.decoders[task].load_state_dict(filtered_decoder_params, strict=False)
    
    return model   
        
        
    
    



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



def get_dataset_few_shot(config):
    
    if config['dataset_name'] == 'NYU':        

        train_transform, val_transform, test_transform = get_transformations(config)

        train_dataset = NYUDatasetFewshot(config = config, data_dir = config['data_dir_NYU'], set = 'train', transform= train_transform)

        val_dataset = NYUDatasetFewshot(config = config, data_dir = config['data_dir_NYU'], set = 'val', transform= val_transform)
        
        test_dataset = NYUDatasetFewshot(config = config, data_dir = config['data_dir_NYU'], set = 'test', transform= test_transform)
        
    elif config['dataset_name'] == 'celebA':
        
        train_transform, val_transform, test_transform = get_transformations(config)
        
        train_dataset = CelebMaskHQDatasetFewshot(config = config, set = 'train', transform=train_transform)
        
        val_dataset = CelebMaskHQDatasetFewshot(config = config, set = 'val', transform=val_transform)
        
        test_dataset = CelebMaskHQDatasetFewshot(config = config, set = 'test', transform=test_transform)
        
        # train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset,[0.6,0.2,0.2])



    else:
        print(" invalid dataset name")
    return train_dataset, val_dataset, test_dataset



def collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))   
    return torch.utils.data.dataloader.default_collate(batch)



def get_dataloader_few_shot(config,train_dataset,val_dataset,test_dataset):   

    train_loader = DataLoader(dataset = train_dataset, batch_size = config['train_batch_size'], shuffle = True, num_workers = config['num_workers'], collate_fn=collate_fn, drop_last=True)   
    
    val_loader = DataLoader(dataset = val_dataset, batch_size = config['val_batch_size'], shuffle = False, num_workers = config['num_workers'],collate_fn=collate_fn, drop_last=True)
        
    test_loader = DataLoader(dataset = test_dataset, batch_size = config['test_batch_size'], shuffle = False, num_workers = config['num_workers'],collate_fn=collate_fn, drop_last=True)

    return train_loader, val_loader, test_loader



class MTLTrainer:
    def __init__(self, config, model, device):    
        self.device = device
        self.config = config
        self.model = model

        for params in self.model.parameters():
            params.requires_grad = True        
        
        self.criterion = get_criterion(self.config)
        self.criterion.to(self.device)    
        print('criterion:', self.criterion)        
        self.train_dataset, self.val_dataset, self.test_dataset = get_dataset_few_shot(self.config)
        self.train_loader, self.val_loader, self.test_loader = get_dataloader_few_shot(self.config, self.train_dataset,self.val_dataset, self.test_dataset)
        
        print('train :', len(self.train_loader))
        print('val :', len(self.val_loader))
        print('test :', len(self.test_loader))
        
        from utils import ModelWrapper
        self.model_wrapper = ModelWrapper(self.config, self.criterion)
        for param in self.model.parameters():
            param.grad = torch.zeros_like(param)       
        
        if config['comb_loss'] == 'uncertainity':                   
            self.log_vars = {task: nn.Parameter(torch.tensor(1.0, device = self.device, requires_grad = True,dtype=torch.float64))  for task in self.config['task_list']}           
            
            for task , value in self.log_vars.items():
                self.model.register_parameter(task, value)            
        else:
            self.log_vars = None
            
        # self.optimizer_decoders = {}    
        # for task in config['task_list']:
        #     self.optimizer_decoders[task] = get_task_optimizer(self.config, self.model.decoders[task])
            
        self.optimizer = get_task_optimizer(self.config, self.model)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, factor=0.1, patience=5, min_lr=0.0000001,verbose= True)
        # initialize the early_stopping object
        self.early_stopping = EarlyStopping(patience=self.config['earlystop_patience'], verbose=True, path=self.config['fname']+'/'+ 'checkpoint.pt')
        
        self.task_early_stopping = {}
        # initialize the task early_stopping object
        for task in self.config['task_list']:
            self.task_early_stopping[task] = EarlyStopping(patience=self.config['task_earlystop_patience'], verbose=True, path=self.config['fname']+'/'+ task+'_checkpoint.pt')
        self.config['es_loss'] = {}
        
        

    def train(self, epoch):  
        
        self.model.train()
        losses =  collections.defaultdict(list)
        metric =  collections.defaultdict(lambda: collections.defaultdict(list))        
        with torch.set_grad_enabled(True):       
            for i, batch in enumerate(tqdm(self.train_loader)):   
                
                images = batch['image'].to(self.device)
                targets = {task: val.to(self.device) for task, val in batch['targets'].items()} 
                
                if self.config['setup'] == 'singletask':
                    output, targets, loss_dict = self.model_wrapper.single_task_forward(self.model,images, targets, 'train')    
                elif self.config['setup'] == 'multitask':
                    output, targets, loss_dict = self.model_wrapper.multi_task_forward(self.model, images, targets, 'train')                  

                                      
                self.optimizer.zero_grad()                           
                loss_dict['total'].backward()
                self.optimizer.step()             
                   
                for task in self.config['task_list']:
                    param = getattr(self.model, task)
                    wandb.log({f"log_vars/{task}":param.item()})
                
                metric_dict = evaluation_metrics(self.config['task_list'],self.config, output, targets)
                for task,value in metric_dict.items():
                    for k,v in value.items():
                        metric[task][k].append(v.cpu().numpy() if isinstance(v, torch.Tensor) else v)
                for keys, value in loss_dict.items():
                    losses[keys].append(value.detach().cpu().numpy())                          

                     
            losses_ = {task: np.mean(val) for task, val in losses.items()}     
            metric_ = {task: {m: np.mean(val) for m, val in values.items()} for task, values in metric.items()}           
            
            
            # torch.save(self.model, '/proj/ltu_mtl/users/x_ricup/MTL_adaptive_results/inter/multi_task_model.pt')
            return losses_,metric_


    def validate(self, epoch):
        self.model.eval()
        losses =  collections.defaultdict(list)
        metric =  collections.defaultdict(lambda: collections.defaultdict(list))
        with torch.no_grad():
            for i, batch in enumerate(tqdm(self.val_loader)):
                images = batch['image'].to(self.device)
                targets = {task: val.to(self.device) for task, val in batch['targets'].items()} 
                if self.config['setup'] == 'singletask':                    
                    output, targets, loss_dict = self.model_wrapper.single_task_forward(self.model,images, targets,'test')    
                elif self.config['setup'] == 'multitask':                    
                    output, targets, loss_dict = self.model_wrapper.multi_task_forward(self.model,images, targets, 'test')
                     
                metric_dict = evaluation_metrics(self.config['task_list'],self.config, output, targets) 
                for task,value in metric_dict.items():
                    for k,v in value.items():
                        metric[task][k].append(v.cpu().numpy() if isinstance(v, torch.Tensor) else v)
                for keys, value in loss_dict.items():
                    losses[keys].append(value.detach().cpu().numpy())      

            losses_ = {task: np.mean(val) for task, val in losses.items()}     
            metric_ = {task: {m: np.mean(val) for m, val in values.items()} for task, values in metric.items()}            
            self.scheduler.step(losses_['total'])            
            self.early_stopping(losses_['total'], self.model, epoch, self.optimizer, model_checkpoint=True, log_vars = self.log_vars, lambda_param = 0)
            if (self.early_stopping.early_stop == True):   
                print("Early stopping")
                return 'STOP', losses_,metric_ , self.model                               
        return 'CONTINUE', losses_, metric_ , self.model

    def test(self, epoch, model):
        epoch= 0
        self.model.eval()
        losses =  collections.defaultdict(list)
        metric =  collections.defaultdict(lambda: collections.defaultdict(list))
        with torch.no_grad():
            for i, batch in enumerate(tqdm(self.test_loader)):
                images = batch['image'].to(self.device)
                targets = {task: val.to(self.device) for task, val in batch['targets'].items()} 
                # if self.config['setup'] == 'singletask':                    
                #     output, targets, loss_dict = self.model_wrapper.single_task_forward(model,images, targets, 'test')    
                # elif self.config['setup'] == 'multitask':                    
                output, targets, loss_dict = self.model_wrapper.multi_task_forward(model,images, targets, 'test') 
                metric_dict = evaluation_metrics(self.config['task_list'],self.config, output, targets) 
                for task,value in metric_dict.items():
                    for k,v in value.items():
                        metric[task][k].append(v.cpu().numpy() if isinstance(v, torch.Tensor) else v)
                for keys, value in loss_dict.items():
                    losses[keys].append(value.detach().cpu().numpy())                
            losses_ = {task: np.mean(val) for task, val in losses.items()}     
            metric_ = {task: {m: np.mean(val) for m, val in values.items()} for task, values in metric.items()}            
                
                
        return losses_,metric_ 


class MultiTaskModel_hooks(nn.Module):
    def __init__(self, backbone: nn.Module, decoders: nn.ModuleDict, tasks: list, config: dict):
        super(MultiTaskModel_hooks, self).__init__()
        assert(set(decoders.keys()) == set(tasks))
        self.backbone = backbone
        self.decoders = decoders
        self.tasks = tasks
        self.hook_outputs = {}

        def hook_fn(module, input, output, task_name):
            self.hook_outputs[task_name] = output

        # Register hooks
        self.hooks = []
        for task in tasks:
            # if task == 'task1':
            #     layer = self.backbone.layer3[2].conv1  # Example layer for task1
            # elif task == 'task2':
            #     layer = self.backbone.layer4[0].conv1  # Example layer for task2
            # ... and so on for other tasks ...
            temp_hook = config['inter_hooks'][task]
            layer = eval('self.backbone.'+ temp_hook)
            hook = layer.register_forward_hook(lambda module, input, output, t=task: hook_fn(module, input, output, t))
            self.hooks.append(hook)

    def forward(self, x):
            # First, perform a forward pass through the backbone to trigger the hooks
        _ = self.backbone(x)

        # Now, use the outputs captured by the hooks for each task
        task_outputs = {}
        for task in self.tasks:
            if task in self.hook_outputs:
                # Use the output from the hook as input to the decoder of the current task
                hook_output = self.hook_outputs[task]
                task_outputs[task] = self.decoders[task](hook_output)
            else:
                # Handle the case where the hook output is not available (optional)
                # You might want to log a warning or use a default process
                # task_outputs[task] = None  # Or some default value
                raise Exception('task hook not specified')

        return task_outputs



    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []



def get_backbone(config,num_input_ch):
    """ Return the backbone """   
    
    num_input_ch = config['input_img_channels']    
    
    if config['backbone'] == 'resnetd50':
        from models.resnetd import ResNetD
        backbone = ResNetD('50', num_input_ch)
    elif config['backbone'] == 'resnetd101':
        from models.resnetd import ResNetD
        backbone = ResNetD('101', num_input_ch)        
    
    else:
        print('backbone does not exist')
    
    bb_channels = backbone.channels    
    
    for ct, child in enumerate(backbone.children()): 
        for param in child.parameters():
            param.requires_grad = True   
        
    return backbone, bb_channels


def get_head(config, bb, task):
    from models.ASPP import DeepLabHead
    from models.class_head import ClassificationHead
    
    # print(bb)
    temp_layer_name = eval('bb.'+ config['inter_hooks'][task])
    in_channels = temp_layer_name.out_channels
    
    print(task, ':', in_channels)
    if config['dataset_name'] == 'NYU':
        config['NYU_num_classes']= {'segmentsemantic': config['seg_classes_NYU'], 'depth_euclidean': 1,'surface_normal': 3, 'edge_texture': 1 }   ### 41
        
        Head =  DeepLabHead(in_channels = in_channels,num_classes=config['NYU_num_classes'][task])
        for params in Head.parameters():
            params.requires_grad = True 
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
        
    backbone, bb_channels = get_backbone(config, config['num_input_ch'])           
    heads = torch.nn.ModuleDict({task: get_head(config, backbone, task) for task in config['task_list']})      
    model = MultiTaskModel_hooks(backbone, heads, config['task_list'], config)           

    return model 


def flatten_dict(d, parent_key='', sep='_'):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, abc.MutableMapping):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

def save_dict_to_csv(all_avg_dict, output_file):
    flattened_dict = flatten_dict(all_avg_dict)
    df = pd.DataFrame(list(flattened_dict.items()), columns=['key', 'mean_std'])
    # df.to_csv(output_file, index=False)
    df_transposed = df.set_index('key').T
    df_transposed.to_csv(output_file, index=False)

def mean_std(data):
    mean = np.mean(data)
    std = np.std(data)
    result = f"{mean:.4f} ± {std:.4f}"
    return result

####################################################################################
#####################################################################################

def main():
    global device 
    device = 'cuda' if torch.cuda.is_available() else 'cpu'    
    config = create_config( args.config_exp) 
    config['device'] = device
    print('Experiment_name: ', config['Experiment_name'])    
    config['setup'] = 'singletask' if len(config['task_list']) == 1 else 'multitask'
    
    
    all_metric_dict = collections.defaultdict(lambda: collections.defaultdict(list))    
    
    for trial in range(config['num_trials']):
    
        fname = config['checkpoint_folder'] + config['dataset_name']+ '/'+ config['backbone']+ '/' + config['Experiment_name']+ '_trial_'+ str(trial)
        config['fname'] = fname
        if not os.path.exists(fname):
            os.makedirs(fname)
        else:
            print('folder already exist')
            
        runs = wandb.init(project= 'Intermediate_features',name=config['Experiment_name']+ '_trial_'+ str(trial), entity='ricupa', config=config, dir = fname, reinit=True)
        wandb.config.update(config, allow_val_change=True)
        
        if config['checkpoint'] == True:
            checkpoint = torch.load(config['checkpoint_folder'] + 'checkpoint.pt')
            model = checkpoint['model'] 
            start_epoch = checkpoint['epoch']+1        
            f = open(fname+  '/validation_dict.json')
            validation_dict = json.load(f)
            f = open(fname+  '/training_dict.json')
            training_dict = json.load(f)
            # epochwise_train_losses_dict = training_dict['loss']
            # epochwise_train_metric_dict = training_dict['metric']
            # epochwise_val_losses_dict = validation_dict['loss']
            # epochwise_val_metric_dict = validation_dict['metric']
            
        else:
            model = get_model(config)   ##### write a get_model function in utils_common
            start_epoch = 0  
            epochwise_train_losses_dict = collections.defaultdict(list)
            epochwise_val_losses_dict = collections.defaultdict(list)
            epochwise_train_metric_dict = collections.defaultdict(lambda: collections.defaultdict(list))   # for nested dictionary
            epochwise_val_metric_dict = collections.defaultdict(lambda: collections.defaultdict(list))
            training_dict = {}
            validation_dict = {}
        
        
        #### save the configs for future use 
        with open(fname+'/'+'config_file.yaml','w')as file:
            doc = yaml.dump(config,file)
            
        model = model.to(device)
        
        model = get_pretrained_weights(config, model)
        print('pretrained meta parameters loaded')
        # for name, params in model.named_parameters():
        #     print(name)
        
        trainer = MTLTrainer(config, model, device)
        
        for epoch in range(start_epoch, config['epochs']):
            print(colored('Epoch %d/%d' %(epoch, config['epochs']-1), 'yellow'))
            print(colored('-'*10, 'yellow'))

            
            loss, metric = trainer.train(epoch)  ##### train the model
            print('train loss: ', loss)
            print('train metric: ', metric)
            
                    
            for key, value in loss.items():
                wandb.log({f"train/loss/{key}": value})
                epochwise_train_losses_dict[key].append(float(value))

            
            for key, value in metric.items():
                if isinstance(value, dict):
                    for sub_key, sub_value in value.items():
                        wandb.log({f"train/metric/{key}/{sub_key}": sub_value})
                        epochwise_train_metric_dict[key][sub_key].append(float(sub_value))
                else:
                    wandb.log({f"train/metric/{key}": value})
                    epochwise_train_metric_dict[key].append(float(value))
            
            ##### validate the model 
            status, vloss, vmetric, model = trainer.validate(epoch)  #### validate the model
            print('val loss: ', vloss)
            print('val metric: ', vmetric)
            
            
            for key, value in vloss.items():
                wandb.log({f"validation/loss/{key}": value})
                epochwise_val_losses_dict[key].append(float(value))
                
            for key, value in vmetric.items():
                if isinstance(value, dict):
                    for sub_key, sub_value in value.items():
                        wandb.log({f"validation/metric/{key}/{sub_key}": sub_value})
                        epochwise_val_metric_dict[key][sub_key].append(float(sub_value))
                else:
                    wandb.log({f"validation/metric/{key}": value})
                    epochwise_val_metric_dict[key].append(float(value))

            
            #### also test for every epoch to see the performance 
            test_loss, test_metric = trainer.test(epoch, model)
            
            print('test loss: ', test_loss)
            print('test metric: ', test_metric)
            
            for key, value in test_loss.items():
                wandb.log({f"test/loss/{key}": value})
                
            for key, value in test_metric.items():
                if isinstance(value, dict):
                    for sub_key, sub_value in value.items():
                        wandb.log({f"test/metric/{key}/{sub_key}": sub_value})
                else:
                    wandb.log({f"test/metric/{key}": value})
            
                                
            if status == 'STOP':
                break
        
        
        ####################################################
        #################################################
        
        print('-----testing the best model -------')
        model = get_model(config)
        model = model.cuda()
        
        if config['comb_loss'] == 'uncertainity':                   
            log_vars = {task: nn.Parameter(torch.tensor(1.0, device = device, requires_grad = True,dtype=torch.float64))  for task in config['task_list']}           
            
            for task , value in log_vars.items():
                model.register_parameter(task, value)
                
        epoch = 0    
        checkpoint = torch.load(fname + '/checkpoint.pt')
        print('load the checkpoint')
        model.load_state_dict(checkpoint['model'], strict = False)
        test_loss, test_metric = trainer.test(epoch, model)
        print('avg_test_metric:', test_metric)        

        #### make sure values are float for saving in json othrwise it gives an error 
        for key, value in test_metric.items():
            if isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    test_metric[key][sub_key] = float(sub_value)                
            else:
                test_metric[key] = float(value)        
        # test_dict['metric'] = test_metric    
        with open(fname+"/"+"test_dict.json", "w") as outfile:
            json.dump(test_metric, outfile)    
        print('test metric dictionary saved as json')   


        
        for key, value in test_metric.items():
            if isinstance(value, dict):
                for sub_key, sub_value in value.items():                        
                    all_metric_dict[key][sub_key].append(float(sub_value))
            else:
                all_metric_dict[key]["value"].append(float(value))
    
    
    # print(all_metric_dict)
    
    all_avg_dict = collections.defaultdict(lambda: collections.defaultdict(list)) 
    
    for key,value in all_metric_dict.items():
        if isinstance(value, dict):
            for sub_key, sub_value in value.items(): 
                print(key, sub_key, mean_std(sub_value))
                all_avg_dict[key][sub_key] = mean_std(sub_value)
        else:
            print(key, mean_std(value))
            all_avg_dict[key] = mean_std(value)
        
    print('overall :', all_avg_dict)
    
    save_dict_to_csv(all_avg_dict, fname +'/'+ 'results.csv')



if __name__ == "__main__":
    main()


