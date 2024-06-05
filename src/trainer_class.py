import torch
import os
from utils import *
import time
from random import sample as SMP
from torch import nn, Tensor
import torch.nn.functional as F
import wandb
import collections
from torchvision.utils import make_grid
import random
from tqdm import tqdm
from pytorchtools import EarlyStopping
from sparsity_early_stop import EarlyStoppingOnStabilization

def prune_input_channels(model, percentile=10):
    """
    Prune the bottom x% of input channels based on their L1 norm.    
    :param model: model
    :param percentile: Percentile threshold for pruning (default 10%)
    :return: modified model
    """
    with torch.no_grad():
        for name, module in model.named_modules():
            if isinstance(module, nn.Conv2d):
                # Compute the L1 norm for each input channel
                l1_norms = module.weight.data.norm(p=1, dim=[0, 2, 3])
                
                # Determine the threshold for pruning based on the given percentile
                threshold = torch.quantile(l1_norms, q=percentile/100)
                
                # Create a mask for channels above the threshold
                keep_channels = l1_norms > threshold
                
                # Prune input channels
                new_in_channels = keep_channels.sum().item()
                if new_in_channels < module.in_channels:
                    module.in_channels = new_in_channels
                    module.weight.data = module.weight.data[:, keep_channels, :, :]
                    
                    # No changes to bias since it's related to output channels    
    return model




def check_zero_params(model):
    for name, param in model.named_parameters():
        if 'weight' in name:
            # Check if any of the filter-wise weights are non-zero
            if torch.nonzero(param).size(0) > 0:
                # print(f'{name} has non-zero weights')
                continue
            else:
                print(f'{name} has all zero weights')
                # print(param.shape)

class Trainer:
    def __init__(self, config, model, device):    
        self.device = device
        self.config = config
        self.model = model

        for params in self.model.parameters():
            params.requires_grad = True
        
        # self.optimizer = get_task_optimizer(self.config, self.model)    
        
        self.criterion = get_criterion(self.config)
        self.criterion.to(self.device)    
        print('criterion:', self.criterion)
        
        self.train_dataset, self.val_dataset, self.test_dataset = get_dataset(self.config)
        self.train_loader, self.val_loader, self.test_loader = get_dataloader(self.config, self.train_dataset,self.val_dataset, self.test_dataset)
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
        
        #if 'lambda_list' in config:
        #    self.lambda_list = config['lambda_list']
        self.last_checkpoint_sparsity = -1 
        self.last_checkpoint_param_sparsity = -1
        
        self.optimizer = get_BB_optimizer(self.config, self.model)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, factor=0.1, patience=5, min_lr=0.0000001,verbose= True)
        # initialize the early_stopping object
        self.early_stopping = EarlyStopping(patience=self.config['earlystop_patience'], verbose=True, path=self.config['fname']+'/'+ 'checkpoint.pt')
        
        self.task_early_stopping = {}
        # initialize the task early_stopping object
        for task in self.config['task_list']:
            self.task_early_stopping[task] = EarlyStopping(patience=self.config['task_earlystop_patience'], verbose=True, path=self.config['fname']+'/'+ task+'_checkpoint.pt')
        self.config['es_loss'] = {}

        self.sparsity_earlystop = EarlyStoppingOnStabilization(patience = config['sparsity_patience'])
        for params in self.model.parameters():
            params.requires_grad = True
            
            
    # def maybe_save_checkpoint_groupsparsity(self, Group_sparsity, epoch, param_sparsity, lambda_param):
    #     thresholds = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]
    #     # Check if Group_sparsity has passed the next threshold
    #     save_condition = any(thresh > self.last_checkpoint_sparsity and Group_sparsity >= thresh for thresh in thresholds)
    #     if save_condition:
    #         log_vars = {}
    #         for task in self.config['task_list']:
    #             log_vars[task] = getattr(self.model, task)
    #         state = {
    #             'epoch': epoch, 
    #             'model': self.model.state_dict(), 
    #             'log_vars': log_vars, 
    #             'lambda_param': lambda_param,
    #             'group_sparsity': Group_sparsity, 
    #             'parameter_sparsity': param_sparsity
    #         }
    #         group_sparsity_str = "{:.2f}".format(Group_sparsity).replace('.', '_')
    #         path = self.config['fname'] + '/' + f'checkpoint_group_sparsity_{group_sparsity_str}.pt'
    #         torch.save(state, path) 
    #         # Update the last checkpointed sparsity level
    #         self.last_checkpoint_sparsity = Group_sparsity
    #         return path 

    #     return None
    
    
    def maybe_save_checkpoint_groupsparsity(self, Group_sparsity, epoch, param_sparsity, lambda_param):
        thresholds = [0, 5, 10,15, 20,25, 30,35, 40,45, 50, 55,60,65, 70, 75, 80]
        
        # If the attribute doesn't exist, initialize it
        if not hasattr(self, 'saved_group_thresholds'):
            self.saved_group_thresholds = set()

        # Check if Group_sparsity has passed a threshold that hasn't been saved yet
        save_condition = any(thresh not in self.saved_group_thresholds and Group_sparsity >= thresh for thresh in thresholds)
        
        if save_condition:
            log_vars = {}
            for task in self.config['task_list']:
                log_vars[task] = getattr(self.model, task)
            state = {
                'epoch': epoch, 
                'model': self.model.state_dict(), 
                'log_vars': log_vars, 
                'lambda_param': lambda_param,
                'group_sparsity': Group_sparsity, 
                'parameter_sparsity': param_sparsity
            }
            group_sparsity_str = "{:.2f}".format(Group_sparsity).replace('.', '_')
            path = self.config['fname'] + '/' + f'checkpoint_group_sparsity_{group_sparsity_str}.pt'
            torch.save(state, path) 
            
            # Update the set of saved group sparsity thresholds
            self.saved_group_thresholds.update({thresh for thresh in thresholds if Group_sparsity >= thresh})

            return path 

        return None

    
    
    def maybe_save_checkpoint_paramparsity(self, Group_sparsity, epoch, param_sparsity, lambda_param):
        thresholds = [0, 5, 10,15, 20,25, 30,35, 40,45, 50, 55,60,65, 70, 75, 80]
        if not hasattr(self, 'saved_thresholds'):
            self.saved_thresholds = set()
        # Check if Group_sparsity has passed the next threshold
        # save_condition = any(thresh > self.last_checkpoint_param_sparsity and param_sparsity >= thresh for thresh in thresholds)
        save_condition = any(thresh not in self.saved_thresholds and param_sparsity >= thresh for thresh in thresholds)
        if save_condition:
            log_vars = {}
            for task in self.config['task_list']:
                log_vars[task] = getattr(self.model, task)
            state = {
                'epoch': epoch, 
                'model': self.model.state_dict(), 
                'log_vars': log_vars, 
                'lambda_param': lambda_param,
                'group_sparsity': Group_sparsity, 
                'parameter_sparsity': param_sparsity
            }    
            param_sparsity_str = "{:.2f}".format(param_sparsity).replace('.', '_')
            path = self.config['fname'] + '/' + f'checkpoint_param_sparsity_{param_sparsity_str}.pt'
            torch.save(state, path)
            # Update the last checkpointed sparsity level
            # self.last_checkpoint_param_sparsity = param_sparsity
            self.saved_thresholds.update({thresh for thresh in thresholds if param_sparsity >= thresh})
            return path 

        return None


    def train(self, epoch):  

        ###### if a list of sparsity parameter  is given to increase or decrease sparsity gradually
        
        # if (self.config['group_sparsity'] == True) and ('lambda_list' in self.config) and (epoch%10 == 0) and (epoch < len(self.config['lambda_list'])*10):
        #     idx = int(epoch/10)
        #     self.config['bb_optimizer_params']['lambda'] = self.config['lambda_list'][idx]
        #     print(epoch, self.config['bb_optimizer_params']['lambda'])
        
        
        # if epoch >= self.config['sparsity_threshold']:
        #     self.optimizer = get_BB_optimizer(self.config, self.model)
    
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
                    output, targets, loss_dict = self.model_wrapper.multi_task_forward(self.model, images, targets,'train')  
                            
                self.optimizer.zero_grad()   
                # self.optimizer_decoders.zero_grad()
                        
                loss_dict['total'].backward()           
                        
                self.optimizer.step()
                
                if (self.config['group_sparsity']) and (self.config['bb_optimizer_params']['penalty']== 'l1'):
                    threshold = 1e-5  # example threshold
                    for name, param in self.model.named_parameters():
                        if 'backbone' in name:
                            param.data *= (param.data.abs() > threshold).float()
                    
                # self.optimizer_decoders.step()
                
                # for name, param in self.model.named_parameters():
                #     print(name, param.requires_grad)
                
                # for name, param in self.model.named_parameters():
                #     if param.grad is None:
                #         print(name, 'grad is none')
                
                # check_zero_params(self.model)
                for task in self.config['task_list']:
                    param = getattr(self.model, task)
                    wandb.log({f"log_vars/{task}":param.item()})
                
                metric_dict = evaluation_metrics(self.config['task_list'],self.config, output, targets)
                for task,value in metric_dict.items():
                    for k,v in value.items():
                        metric[task][k].append(v.cpu().numpy() if isinstance(v, torch.Tensor) else v)
                for keys, value in loss_dict.items():
                    losses[keys].append(value.detach().cpu().numpy())
                    

                if (self.config['dataset_name'] == 'celebA') and (i == 100):
                    break
            
            losses_ = {task: np.mean(val) for task, val in losses.items()}     
            metric_ = {task: {m: np.mean(val) for m, val in values.items()} for task, values in metric.items()}
            
            if self.config['group_sparsity']:
                # sparsity = calculate_percentage_sparsity(self.model)
                # wandb.log({"train/sparsity": sparsity})                                 
                Group_sparsity,param_sparsity,_,_ = calculate_percentage_sparsity_params(self.model)          
                      
                wandb.log({"train/param_sparsity": param_sparsity})        
                wandb.log({"train/group_sparsity": Group_sparsity})


                ##### save intermediate sparse models
                self.cp_path_group_sparsity = self.maybe_save_checkpoint_groupsparsity(Group_sparsity, epoch, param_sparsity, self.config['bb_optimizer_params']['lambda'])
                self.cp_path_param_sparsity = self.maybe_save_checkpoint_paramparsity(Group_sparsity, epoch, param_sparsity, self.config['bb_optimizer_params']['lambda'])

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
                    
                    output, targets, loss_dict = self.model_wrapper.single_task_forward(self.model,images, targets,'val')    
                elif self.config['setup'] == 'multitask':
                    
                    output, targets, loss_dict = self.model_wrapper.multi_task_forward(self.model,images, targets, 'val') 

                metric_dict = evaluation_metrics(self.config['task_list'],self.config, output, targets) 

                for task,value in metric_dict.items():
                    for k,v in value.items():
                        metric[task][k].append(v.cpu().numpy() if isinstance(v, torch.Tensor) else v)
                for keys, value in loss_dict.items():
                    losses[keys].append(value.detach().cpu().numpy())
            
                if (self.config['dataset_name'] == 'celebA') and (i == 50):
                    break
                
                
            losses_ = {task: np.mean(val) for task, val in losses.items()}     
            metric_ = {task: {m: np.mean(val) for m, val in values.items()} for task, values in metric.items()}
            
            self.scheduler.step(losses_['total'])

        # if self.config['group_sparsity']:
        lambda_param = self.config['bb_optimizer_params']['lambda']    
        # else:
        #     lambda_param = None
        
        
        
        if self.config['group_sparsity']:                               
            Group_sparsity,param_sparsity,_,_ = calculate_percentage_sparsity_params(self.model)
            
            if param_sparsity > self.config['sparsity_threshold'] - 10:  #### it if is taking a long time to reach the threshold then it will triger early stop                         
                                        
                self.early_stopping(losses_['total'], self.model, epoch, None, model_checkpoint=True,log_vars = self.log_vars, lambda_param = lambda_param)
                if (self.early_stopping.early_stop == True):   #######(sum(self.config['flag'].values()) == 0 ) or 
                    print("Early stopping")
                    # break   
                    return 'STOP', losses_,metric_ , self.model
                
            if (param_sparsity > self.config['sparsity_threshold']) or (epoch == self.config['epochs']-1) or ((self.sparsity_earlystop.should_stop(param_sparsity))):   ####(param_sparsity > 40) and 
                print('training stopping')  
                log_vars = {}
                for task in self.config['task_list']:
                    log_vars[task] = getattr(self.model, task)
                state = {'epoch': epoch, 'model': self.model.state_dict(), 'log_vars': log_vars, 'lambda_param': lambda_param}
                path=self.config['fname']+'/'+ 'checkpoint.pt'
                torch.save(state, path)
                return 'STOP', losses_, metric_ , self.model    
        # else:
        self.early_stopping(losses_['total'], self.model, epoch, self.optimizer, model_checkpoint=True, log_vars = self.log_vars, lambda_param = lambda_param)
        if (self.early_stopping.early_stop == True):   ##### (sum(self.config['flag'].values()) == 0 ) or 
            print("Early stopping")
            return 'STOP', losses_,metric_ , self.model    
                           
        return 'CONTINUE', losses_, metric_ , self.model


    def test(self, epoch, model):
        epoch= 0
        model.eval()
        losses =  collections.defaultdict(list)
        metric =  collections.defaultdict(lambda: collections.defaultdict(list))
        with torch.no_grad():
            for i, batch in enumerate(tqdm(self.test_loader)):
                images = batch['image'].to(self.device)
                targets = {task: val.to(self.device) for task, val in batch['targets'].items()} 

                if self.config['setup'] == 'singletask':
                    
                    output, targets, loss_dict = self.model_wrapper.single_task_forward(model,images, targets, 'test')    
                elif self.config['setup'] == 'multitask':
                    
                    output, targets, loss_dict = self.model_wrapper.multi_task_forward(model,images, targets, 'test') 

                metric_dict = evaluation_metrics(self.config['task_list'],self.config, output, targets) 

                for task,value in metric_dict.items():
                    for k,v in value.items():
                        metric[task][k].append(v.cpu().numpy() if isinstance(v, torch.Tensor) else v)
                for keys, value in loss_dict.items():
                    losses[keys].append(value.detach().cpu().numpy())
            
                # if i == 0:
                #     break
                
            losses_ = {task: np.mean(val) for task, val in losses.items()}     
            metric_ = {task: {m: np.mean(val) for m, val in values.items()} for task, values in metric.items()}
            
            # self.scheduler.step()
            
            # if (epoch %10 == 0) and (self.config['wandb_img_log'] == True):
            #     #### for visulaization on wandb
            #     img_idx = 5
            #     img = images[img_idx,:,:,:]
            #     size = img.shape[-2:]
            #     wandb.log({'input_image': wandb.Image(img)})
            
            
            #     if 'segmentsemantic' in self.config['task_list']:
            #         target = targets['segmentsemantic'][img_idx,0,:,:]
            #         # target = torch.squeeze(target,0)
            #         out = F.interpolate(output['segmentsemantic'], size=size, mode="bilinear")
            #         out = F.softmax(out, dim = 1)
            #         out = out[img_idx,:,:,:]
            #         out = torch.argmax(out, dim = 0)
                    
            #         if self.config['dataset_name'] == 'Taskonomy':
            #             out = draw_segmentation_map_taskonomy(out).permute(2,0,1).float()
            #             tar = draw_segmentation_map_taskonomy(target).permute(2,0,1).float()
            #         elif self.config['dataset_name'] == 'NYU':
            #             out = draw_segmentation_map_NYU(out).permute(2,0,1).float()
            #             tar = draw_segmentation_map_NYU(target).permute(2,0,1).float()
            #         else:
            #             print('dataset not found')
                        
            #         # image_grid = [torch.tensor(tar), torch.tensor(out)]        
            #         image_grid = [tar, out]         
            #         grid = make_grid(image_grid, nrows=2) 
            #         wandb.log({"Segmentation- GT (left) and pred (right)": wandb.Image(grid)})              
            #         # wandb.log({'segmentation_GT': wandb.Image(tar)})
            #         # wandb.log({'segmentation_pred': wandb.Image(out)})  
                    
                    
                    
            #     if 'depth_euclidean' in self.config['task_list']:
            #         tar = F.interpolate(targets['depth_euclidean'], size=size,mode="bilinear")
            #         tar = tar[img_idx,:,:,:]
            #         # wandb.log({'depth_GT': wandb.Image(tar)})
            #         out = F.interpolate(output['depth_euclidean'], size=size, mode="bilinear")
            #         out = out[img_idx,:,:,:]
            #         # out = out.unsqueeze(0)
                    
            #         # image_grid = [torch.tensor(tar), torch.tensor(out)]   
            #         image_grid = [tar, out]                  
            #         grid = make_grid(image_grid, nrows=2) 
            #         wandb.log({"Depth- GT (left) and pred (right)": wandb.Image(grid)})                 
            #         # wandb.log({'depth_pred': wandb.Image(out)})
                    
            #     if  'edge_texture' in self.config['task_list']:
            #         tar1 = F.interpolate(targets['edge_texture'], size=size, mode="bilinear")
            #         tar1 = tar1[img_idx,:,:,:]
            #         # tar = (tar1>0.5).float()
            #         out1 = F.interpolate(output['edge_texture'], size=size, mode="bilinear")
            #         out1 = out1[img_idx,:,:,:]   
            #         # out = (out1>0.5).float()                  
            #         image_grid = [tar1, out1]               
            #         grid = make_grid(image_grid, nrows=2) 
            #         wandb.log({"Edge - GT (left) and pred (right)": wandb.Image(grid)}) 

            #     if 'surface_normal' in self.config['task_list']:
            #         tar = F.interpolate(targets['surface_normal'], size=size, mode="bilinear")
            #         tar = tar[img_idx,:,:,:]
            #         # wandb.log({'SN_GT': wandb.Image(tar)})                   
            #         out = F.interpolate(output['surface_normal'], size=size, mode="bilinear")
            #         out = out[img_idx,:,:,:]
            #         # wandb.log({'SN_pred': wandb.Image(out)})     
            #         # image_grid = [torch.tensor(tar), torch.tensor(out)]   
            #         image_grid = [tar, out]                  
            #         grid = make_grid(image_grid, nrows=2) 
            #         wandb.log({"Surface_normal - GT (left) and pred (right)": wandb.Image(grid)})
                
                
        return losses_,metric_ 


    
    







            # if (epoch %10 == 0) and (self.config['wandb_img_log'] == True):
            #     #### for visulaization on wandb
            #     img_idx = 5
            #     img = images[img_idx,:,:,:]
            #     size = img.shape[-2:]
            #     wandb.log({'input_image': wandb.Image(img)})
            
            
            #     if 'segmentsemantic' in self.config['task_list']:
            #         target = targets['segmentsemantic'][img_idx,0,:,:]
            #         # target = torch.squeeze(target,0)
            #         out = F.interpolate(output['segmentsemantic'], size=size, mode="bilinear")
            #         out = F.softmax(out, dim = 1)
            #         out = out[img_idx,:,:,:]
            #         out = torch.argmax(out, dim = 0)
                    
            #         if self.config['dataset_name'] == 'Taskonomy':
            #             out = draw_segmentation_map_taskonomy(out).permute(2,0,1).float()
            #             tar = draw_segmentation_map_taskonomy(target).permute(2,0,1).float()
            #         elif self.config['dataset_name'] == 'NYU':
            #             out = draw_segmentation_map_NYU(out).permute(2,0,1).float()
            #             tar = draw_segmentation_map_NYU(target).permute(2,0,1).float()
            #         else:
            #             print('dataset not found')
                        
            #         # image_grid = [torch.tensor(tar), torch.tensor(out)]        
            #         image_grid = [tar, out]         
            #         grid = make_grid(image_grid, nrows=2) 
            #         wandb.log({"Segmentation- GT (left) and pred (right)": wandb.Image(grid)})              
            #         # wandb.log({'segmentation_GT': wandb.Image(tar)})
            #         # wandb.log({'segmentation_pred': wandb.Image(out)})  
                    
                    
                    
            #     if 'depth_euclidean' in self.config['task_list']:
            #         tar = F.interpolate(targets['depth_euclidean'], size=size,mode="bilinear")
            #         tar = tar[img_idx,:,:,:]
            #         # wandb.log({'depth_GT': wandb.Image(tar)})
            #         out = F.interpolate(output['depth_euclidean'], size=size, mode="bilinear")
            #         out = out[img_idx,:,:,:]
            #         # out = out.unsqueeze(0)
                    
            #         # image_grid = [torch.tensor(tar), torch.tensor(out)]   
            #         image_grid = [tar, out]                  
            #         grid = make_grid(image_grid, nrows=2) 
            #         wandb.log({"Depth- GT (left) and pred (right)": wandb.Image(grid)})                 
            #         # wandb.log({'depth_pred': wandb.Image(out)})
                    
            #     if  'edge_texture' in self.config['task_list']:
            #         tar1 = F.interpolate(targets['edge_texture'], size=size, mode="bilinear")
            #         tar1 = tar1[img_idx,:,:,:]
            #         # tar = (tar1>0.5).float()
            #         out1 = F.interpolate(output['edge_texture'], size=size, mode="bilinear")
            #         out1 = out1[img_idx,:,:,:]   
            #         # out = (out1>0.5).float()                  
            #         image_grid = [tar1, out1]               
            #         grid = make_grid(image_grid, nrows=2) 
            #         wandb.log({"Edge - GT (left) and pred (right)": wandb.Image(grid)}) 

            #     if 'surface_normal' in self.config['task_list']:
            #         tar = F.interpolate(targets['surface_normal'], size=size, mode="bilinear")
            #         tar = tar[img_idx,:,:,:]
            #         # wandb.log({'SN_GT': wandb.Image(tar)})                   
            #         out = F.interpolate(output['surface_normal'], size=size, mode="bilinear")
            #         out = out[img_idx,:,:,:]
            #         # wandb.log({'SN_pred': wandb.Image(out)})     
            #         # image_grid = [torch.tensor(tar), torch.tensor(out)]   
            #         image_grid = [tar, out]                  
            #         grid = make_grid(image_grid, nrows=2) 
            #         wandb.log({"Surface_normal - GT (left) and pred (right)": wandb.Image(grid)})
            
            

        






