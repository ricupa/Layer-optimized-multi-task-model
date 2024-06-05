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
from trainer_class import Trainer
import wandb
import json
from proxssi.groups.resnet_gp import resnet_groups
from proxssi.optimizers.adamw_hf import AdamW
# from proxssi.tests import penalties
from proxssi import penalties


# from torch.utils.tensorboard import SummaryWriter
# import json
torch.autograd.set_detect_anomaly(True)

parser = argparse.ArgumentParser(description='Training')
parser.add_argument('--config_exp', help='Config file for the experiment')
args = parser.parse_args()


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

def check_zero_params(model):
    for name, param in model.named_parameters():
        if 'weight' in name:
            # Check if any of the filter-wise weights are non-zero
            if torch.nonzero(param).size(0) > 0:
                # print(f'{name} has non-zero weights')
                continue
            else:
                print(f'{name} has all zero weights')

def main():
    global device 

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)

    config = create_config( args.config_exp) 
    config['device'] = device

    config['setup'] = 'singletask' if len(config['task_list']) == 1 else 'multitask'
     
      
    print('Experiment_name: ', config['Experiment_name'])
    
    all_metric_dict = collections.defaultdict(lambda: collections.defaultdict(list)) 
    
    
    for trial in range(config['num_trials']):

        fname = config['checkpoint_folder'] + config['dataset_name']+ '/'+ config['backbone']+ '/stl/' + config['Experiment_name'] + '_trial_'+ str(trial)
        print(fname)

        config['fname'] = fname
        config['setup'] = 'singletask' if len(config['task_list']) == 1 else 'multitask'
        if not os.path.exists(fname):
            os.makedirs(fname)
        else:
            print('folder already exist')
        
        tempname = 'Intermediate_features_sparsity_'+ config['dataset_name']
        runs = wandb.init(project= tempname, name=config['Experiment_name']+ '_trial_'+ str(trial), entity='ricupa', config=config, dir = fname, reinit=True)
        wandb.config.update(config, allow_val_change=True)

        if config['checkpoint'] == True:
            checkpoint = torch.load(config['checkpoint_folder'] + 'checkpoint.pt')
            model = checkpoint['model'] 
            start_epoch = checkpoint['epoch']+1
            
            f = open(fname+  '/validation_dict.json')
            validation_dict = json.load(f)
            f = open(fname+  '/training_dict.json')
            training_dict = json.load(f)
            epochwise_train_losses_dict = training_dict['loss']
            epochwise_train_metric_dict = training_dict['metric']
            epochwise_val_losses_dict = validation_dict['loss']
            epochwise_val_metric_dict = validation_dict['metric']
            
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
        
        ######## flag for earlystopping the task 
        config['flag'] = {}
        for task in config['task_list']:
            config['flag'][task] = 1

        trainer = Trainer(config, model, device)

   
        for epoch in range(start_epoch, config['epochs']):
            print(colored('Epoch %d/%d' %(epoch, config['epochs']-1), 'yellow'))
            print(colored('-'*10, 'yellow'))

            
            loss, metric = trainer.train(epoch)  ##### train the model
            print('train loss: ', loss)
            print('train metric: ', metric)

            # for keys in loss.keys():
            #     epochwise_train_losses_dict[keys].append(float(loss[keys]))          
            
            # for task,value in metric.items():
            #     for k,v in value.items():
            #         epochwise_train_metric_dict[task][k].append(float(v))

            
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
            
            # wandb_logger(config,epoch,vloss,vmetric,set='validation')
            
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
                    
            # for keys in vloss.keys():
            #     epochwise_val_losses_dict[keys].append(vloss[keys])
            
            # for task,value in vmetric.items():
            #     for k,v in value.items():
            #         epochwise_val_metric_dict[task][k].append(v)   
                    
            
            ##### also test for every epoch to see the performance 
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
            
            # training_dict['train_time'] = train_time
            # training_dict['loss'] = epochwise_train_losses_dict
            # training_dict['metric'] = epochwise_train_metric_dict
            
            # with open(fname+"/"+"training_dict.json", "w") as outfile:
            #     json.dump(training_dict, outfile)
                
            # # validation_dict['val_time'] = inference_time
            # validation_dict['loss'] = epochwise_val_losses_dict
            # validation_dict['metric'] = epochwise_val_metric_dict

        
            # with open(fname+"/"+"validation_dict.json", "w") as outfile:
            #     json.dump(validation_dict, outfile)


################################################################################################
        print('-----testing the best model -------')
        
        
        
        model = get_model(config)
        model = model.cuda()
        epoch = 0
        
        # if config['setup'] == 'singletask':
        #     print('singletask')
        print('load checkpoint')
        checkpoint = torch.load(fname + '/checkpoint.pt')      
        
        print('add log_vars to model')
        log_vars_ = checkpoint['log_vars']    
        log_vars = {task: nn.Parameter(torch.tensor(log_vars_[task], device = device, requires_grad = True,dtype=torch.float64))  for task in config['task_list']}         

        for task , value in log_vars.items():
            model.register_parameter(task, value)

        
        print('add lambda_param to model')
        lambda_param = checkpoint['lambda_param']   
        
        # if config['learnable_lambda'] == True:
        model.register_parameter('lambda_param', nn.Parameter(torch.tensor(lambda_param,device = device, requires_grad = True, dtype=torch.float64)))
        
        model.load_state_dict(checkpoint['model'], strict = False)
        
        test_loss, test_metric = trainer.test(epoch, model)
        print('avg_test_metric:', test_metric)

        Group_sparsity,param_sparsity,_,_ = calculate_percentage_sparsity_params(model)          
          
        wandb.log({"test/param_sparsity": param_sparsity})        
        wandb.log({"test/group_sparsity": Group_sparsity})
        test_metric['group_sparsity'] = Group_sparsity
        test_metric['param-sparsity'] = param_sparsity
        # else:
        #     test_metric = {}
        #     checkpoint = torch.load(fname + '/checkpoint.pt')
        #     model.load_state_dict(checkpoint['model'])
            
        #     for task in config['task_list']:
        #         task_chkpt = fname + '/' + task + '_checkpoint.pt'
        #         if os.path.exists(task_chkpt):
        #             print('task checkpoint exists')
        #             checkpoint = torch.load(task_chkpt)
        #             model.load_state_dict(checkpoint['model'])
        #         else:
        #             print('task checkpoint not found')
                
        #         loss, metric = trainer.test(epoch, model)
        #         test_metric[task] = metric[task]
            
        #     print('avg_test_metric:', test_metric)
                    
        # test_dict['loss'] = np.float64(test_loss)
        
        
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
    
    # save_dict_to_csv(all_avg_dict, 'out_csv_'+ config['dataset_name']+'/' + config['Experiment_name'] + '.csv')
    save_dict_to_csv(all_avg_dict, fname + 'results.csv')


if __name__ == "__main__":
    main()
