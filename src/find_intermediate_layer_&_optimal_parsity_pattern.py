# %%

#### This file has the support code for finding the intermediate layer by visualizing the sparsity patters in the backbone of the trained single task network. 
##### below you may find the code for generating the sparsity pattern for a single trial or finding the average sparsity patterns across multiple trials. 


import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
device = 'cpu'
import yaml
from utils import *
    
# %%
def check_zero_params(model, config):
    flag = []
    layername = []
    for name, param in model.named_parameters():
        if ('backbone' in name) and ('weight' in name) and ('conv' in name) and (len(param.shape) == 4):
            layername.append(name)
            if torch.nonzero(param).size(0) > 0:
                flag.append(1)  ###### print(f'{name} has non-zero weights')
               
                # continue
            else:
                flag.append(0)                
               
    layername = get_layer_name(layername)
    
    flag = np.asarray(flag)
    flag = np.expand_dims(flag,1)
    all_ = np.ones_like(flag)
    data = np.concatenate((all_, flag), axis =1)
    # print(data.shape)
    return data, layername



#### calculate percentage sparsity
def calculate_percentage_sparsity(model):
    num_layer = 0
    num_nonzero = 0
    num_zero = 0
    for name, param in model.named_parameters():
        if ('backbone' in name) and ('weight' in name) and (len(param.shape) == 4):
            
            for i in range(param.shape[1]):
                num_layer += 1
                
                if torch.nonzero(param[:,i,:,:]).size(0) > 0:
                    num_nonzero += 1
                else:
                    num_zero += 1
                    
    assert num_layer == (num_nonzero + num_zero)
    
    spar = (num_zero / num_layer)*100
    
    return spar


def get_layer_name(lname):
    layer_name = []
    for i, name in enumerate(lname):
        temp = name.split('.')
        temp1 = temp[1:-1]
        n = '.'.join(temp1)
        layer_name.append(n)
    return layer_name

def check_zero_param_filterwise(model, config):
    # df = pd.DataFrame()
    layername = []
    _filters = []
    sparse_filters= []
    for name, param in model.named_parameters():
        
        if ('backbone' in name) and ('weight' in name) and ('conv' in name) and (len(param.shape) == 4):
            layername.append(name)
            # print(name, param.shape)
            count = 0
            for i in range(param.shape[1]):
                if torch.nonzero(param[:,i,:,:]).size(0) > 0:
                        ########non-zero values 
                    count += 1
                else:
                    continue
                    
                    
            sparse_filters.append(count)   
            _filters.append(param.shape[1])    
                    
    assert len(sparse_filters) == len(_filters)
    assert len(sparse_filters) == len(layername)
    layername = get_layer_name(layername)
    
    return layername, _filters, sparse_filters 

# %%
##### TO PLOT THE SPARSITY PATTERNS for a single model -----------------------------------------
def plot_sparsity(df, data, lname, task):
    f, axs = plt.subplots(2, figsize=(45,30), gridspec_kw={'wspace':0, 'hspace':0}, facecolor='white')
    
    # Define a colorblind-friendly color palette
    palette = sns.color_palette("colorblind")
    for ax in axs:
        ax.set_facecolor('white')
    
    # First plot with bar charts for layer sparsity
    sns.barplot(x=df["layername"], y=df["filter_length"], color=palette[0], ax=axs[0], label='without sparsity')
    sns.barplot(x=df["layername"], y=df["sparse_filters"], color=palette[3], ax=axs[0], label='with sparsity' + ' (tasks: ' + task + ')')
    legend=axs[0].legend(loc="upper left", frameon=True, fontsize=20,facecolor='white')
    for text in legend.get_texts():
        text.set_color('black')
    axs[0].tick_params(axis='x', rotation=90, labelsize=18)
    axs[0].tick_params(colors='black')  # Ensure tick colors are black
    axs[0].set(xlabel="", ylabel="")
    # Set the colors of labels to black, if not already
    axs[0].xaxis.label.set_color('black')
    axs[0].yaxis.label.set_color('black')
    
    # For the plot spines
    for spine in axs[0].spines.values():
        spine.set_edgecolor('black')

    # Second plot with the spy matrix
    axs[1].spy(np.transpose(data), markersize=20,color = 'blue')
    axs[1].set_xticks(np.arange(len(data)))
    axs[1].set_xticklabels(labels=lname, rotation=90, size=18)
    axs[1].set_yticks(np.arange(2))
    axs[1].set_yticklabels(labels=['ResNet50', 'Sparse'])
    axs[1].tick_params(colors='black')  # Ensure tick colors are black for the second plot as well
    
    
    # plt.style.use('default')
    # p = 'pattern_plots/celebA_res50/'
    # f.savefig(p+'eyebrow_2.png',facecolor=f.get_facecolor(), edgecolor='none', bbox_inches='tight')
    
    plt.show()

dir_checkpoint = "/proj/ltu_mtl/users/x_ricup/MTL_adaptive_results/meta/NYU/Exp1_single_seg_sparse1e_3_trial_4"
# dir_checkpoint = "/proj/ltu_mtl/users/x_ricup/MTL_adaptive_results/meta/celebA/5_single_class_smile_l1l2_1e_4_trial_2"


config = create_config(dir_checkpoint +'/config_file.yaml') 
# config['task_list'] = ['segmentsemantic', 'depth_euclidean']
model = get_model(config)

# model = model.cuda()
checkpoint = torch.load(dir_checkpoint + '/checkpoint.pt', map_location= device) ### checkpoint.pt
model.load_state_dict(checkpoint['model'], strict = False)     
data, lname = check_zero_params(model, config)
layername, _filters, sparse_filters = check_zero_param_filterwise(model, config)
df = pd.DataFrame({'layername': layername, 'filter_length': _filters, 'sparse_filters':sparse_filters})
task = "+".join(config['task_list'])
plot_sparsity(df, data, lname, task)


# %% MAKE AVERAGE SPARSITY PATTERN PLOTS ACROSS MULTIPLE TRIALS, in this work we made the plots considering 5 trials 


def get_minimum_sparsity_pattern(trials, dir_checkpoint):
    min_pattern = None

    for i in range(trials):
        # print('trial -', i)
        exp_name = dir_checkpoint+ str(i)
        config = create_config(exp_name +'/config_file.yaml') 
        model = get_model(config)
        checkpoint = torch.load(exp_name + '/checkpoint.pt', map_location= device) ### checkpoint.pt
        model.load_state_dict(checkpoint['model'], strict = False)
        
        layername, _filters, sparse_filters = check_zero_param_filterwise(model, config)
        # print('-------')
        # print(_filters)
        # print(sparse_filters)
        # Convert to binary sparsity matrix (1 for active filters, 0 for sparse)
        # binary_matrix = [1 if count > 0 else 0 for count in sparse_filters]
        binary_matrix = [1 if sparse_filters[i] > 0.01 * _filters[i] else 0 for i in range(len(sparse_filters))]

        

        if min_pattern is None:
            min_pattern = binary_matrix
        else:
            # Element-wise multiplication to update the minimum sparsity pattern
            min_pattern = [min_pattern[i] + binary_matrix[i] for i in range(len(binary_matrix))]
            # print(min_pattern)
            
    min_pattern = [divmod(x,trials)[0] for x in min_pattern]
    
    # print(min_pattern)
    return min_pattern, layername, _filters, config

# Example usage
# dir_checkpoint = "/proj/ltu_mtl/users/x_ricup/MTL_adaptive_results/meta/NYU/Exp3_single_sn_sparse1e_4_trial_"
dir_checkpoint = '/proj/ltu_mtl/users/x_ricup/MTL_adaptive_results/meta/celebA/5_single_class_smile_l1l2_1e_4_trial_'
#### Exp1_single_seg_sparse1e_3_trial_
#### Exp2_single_depth_sparse1e_3_trial_
### Exp3_single_sn_sparse1e_3_trial_
### Exp4_single_edge_sparse1e_3_trial_
#### for resnet 100
## Exp1_single_seg_res100_1e_3_trial_
## Exp2_single_depth_res100_1e_3_trial_
## Exp3_single_sn_res100_1e_3_trial_
## Exp4_single_edge_res100_1e_3_trial_
### for celeb resnet 50 
## 1_single_segmentsemantic_l1l2_1e_3_trial_
## 2_single_class_lipstick_1e_3_trial_
## 5_single_class_smile_l1l2_1e_3_trial_
## 7_single_class_biglips_l1l2_1e_3_trial_
## 6_single_class_highcheekbones_l1l2_1e_3_trial_
## 3_single_class_male_1e_3_trial_
## 4_single_class_eyebrows_1e_3_trial_



trials = 3
minimum_sparsity_pattern, lname, _filters, config = get_minimum_sparsity_pattern(trials, dir_checkpoint)

# Assuming minimum_sparsity_pattern and _filters are lists of the same length
active_channels = [min_pattern * total_filters for min_pattern, total_filters in zip(minimum_sparsity_pattern, _filters)]


if config['backbone'] == 'resnetd50':
    f, axs = plt.subplots(1, figsize =(30,15), gridspec_kw={'wspace':0, 'hspace':0},facecolor='white')
else:
    f, axs = plt.subplots(1, figsize =(40,10), gridspec_kw={'wspace':0, 'hspace':0}, facecolor='white')
palette = sns.color_palette("colorblind")

f.set_facecolor('white')
axs.set_facecolor('white')

temp_data = np.column_stack((_filters, active_channels))

axs.spy(np.transpose(temp_data), markersize=20,color = 'blue')
axs.set_xticks(np.arange(len(temp_data)))
axs.set_xticklabels(labels=lname, rotation=90, size=18)
axs.set_yticks(np.arange(2))
axs.set_yticklabels(labels=['ResNet50', 'Sparse'])
axs.tick_params(colors='black') 

# sns.barplot(x=layername, y=_filters, color=palette[0], ax=axs, label='without sparsity')
# sns.barplot(x=layername, y=active_channels, color=palette[3], ax=axs, label='with sparsity')      ##+ '(task: '+ config['task_list']+ ')'
# axs.tick_params(axis='x', rotation=90, labelsize=18)
# axs.legend(loc="upper left", frameon=False, fontsize=24)

axs.set_facecolor('white')  # Ensure the axes background is also white
axs.tick_params(colors='black')  # Set the color of the tick marks to black
axs.xaxis.label.set_color('black')  # Set the color of the x-axis label to black
axs.yaxis.label.set_color('black')  # Set the color of the y-axis label to black
axs.title.set_color('black')  # Set the color of the title to black
for spine in axs.spines.values():  # Set the color of the plot spines to black
    spine.set_edgecolor('black')

#%% saving the plot
plt.style.use('default')
p = 'pattern_plots_inter/celebA_res50/'
f.savefig(p+'smile_avg_1.png',facecolor=f.get_facecolor(), edgecolor='none', bbox_inches='tight')


# %%
### CREATE A MODEL WITH MULTIPLE EXITS FOR DIFFERNT TASKS AND THEN VISUALIZA IT 

# class MultiTaskModel_hooks(nn.Module):
#     def __init__(self, backbone: nn.Module, decoders: nn.ModuleDict, tasks: list, config: dict):
#         super(MultiTaskModel_hooks, self).__init__()
#         assert(set(decoders.keys()) == set(tasks))
#         self.backbone = backbone
#         self.decoders = decoders
#         self.tasks = tasks
#         self.hook_outputs = {}

#         def hook_fn(module, input, output, task_name):
#             self.hook_outputs[task_name] = output

#         # Register hooks
#         self.hooks = []
#         for task in tasks:
#             # if task == 'task1':
#             #     layer = self.backbone.layer3[2].conv1  # Example layer for task1
#             # elif task == 'task2':
#             #     layer = self.backbone.layer4[0].conv1  # Example layer for task2
#             # ... and so on for other tasks ...
#             temp_hook = config['inter_hooks'][task]
#             layer = eval('self.backbone.'+ temp_hook)
#             hook = layer.register_forward_hook(lambda module, input, output, t=task: hook_fn(module, input, output, t))
#             self.hooks.append(hook)

#     def forward(self, x):
#             # First, perform a forward pass through the backbone to trigger the hooks
#         _ = self.backbone(x)

#         # Now, use the outputs captured by the hooks for each task
#         task_outputs = {}
#         for task in self.tasks:
#             if task in self.hook_outputs:
#                 # Use the output from the hook as input to the decoder of the current task
#                 hook_output = self.hook_outputs[task]
#                 task_outputs[task] = self.decoders[task](hook_output)
#             else:
#                 # Handle the case where the hook output is not available (optional)
#                 # You might want to log a warning or use a default process
#                 # task_outputs[task] = None  # Or some default value
#                 raise Exception('task hook not specified')

#         return task_outputs



#     def remove_hooks(self):
#         for hook in self.hooks:
#             hook.remove()
#         self.hooks = []



# def get_backbone(config,num_input_ch):
#     """ Return the backbone """   
    
#     num_input_ch = config['input_img_channels']    
    
#     if config['backbone'] == 'resnetd50':
#         from models.resnetd import ResNetD
#         backbone = ResNetD('50', num_input_ch)
#     elif config['backbone'] == 'resnetd101':
#         from models.resnetd import ResNetD
#         backbone = ResNetD('101', num_input_ch)        
#     elif config['backbone'] == 'vit':
#         backbone = ViTBackbone()
#     else:
#         print('backbone does not exist')
    
#     bb_channels = backbone.channels    
    
#     for ct, child in enumerate(backbone.children()): 
#         for param in child.parameters():
#             param.requires_grad = True   
        
#     return backbone, bb_channels


# def get_head(config, bb, task):
#     from models.ASPP import DeepLabHead
#     from models.class_head import ClassificationHead
    
#     # print(bb)
#     temp_layer_name = eval('bb.'+ config['inter_hooks'][task])
#     in_channels = temp_layer_name.out_channels
    
#     print(task, ':', in_channels)
#     if config['dataset_name'] == 'NYU':
#         config['NYU_num_classes']= {'segmentsemantic': config['seg_classes_NYU'], 'depth_euclidean': 1,'surface_normal': 3, 'edge_texture': 1 }   ### 41
        
#         Head =  DeepLabHead(in_channels = in_channels,num_classes=config['NYU_num_classes'][task])
#         for params in Head.parameters():
#             params.requires_grad = True 
#     elif config['dataset_name'] == 'celebA':
#         if task == 'segmentsemantic':
#             Head = DeepLabHead(in_channels = in_channels,num_classes=3)      ######0/255:'background',1:'hair',2:'skin'
#         else:
#             Head = ClassificationHead(in_channels = in_channels, num_classes = 1)     ### binary classifications        
#         for params in Head.parameters():
#             params.requires_grad = True  
        
#     else:
#         raise NotImplementedError('Task head for the dataset not found')

#     return Head    


# def get_model(config):
        
#     backbone, bb_channels = get_backbone(config, config['num_input_ch'])    
#     # if config['setup'] == 'singletask':
#     #     task = config['task_list'][0]
#     #     head = get_head(config, bb_channels[-1], task)
#     #     model = SingleTaskModel(backbone, head, task)
        
#     # elif config['setup'] == 'multitask':        
#     heads = torch.nn.ModuleDict({task: get_head(config, backbone, task) for task in config['task_list']})      
#     model = MultiTaskModel_hooks(backbone, heads, config['task_list'], config)           

#     return model 

# # %%
# from torchviz import make_dot
# config = create_config('config_berzelius/exp_multi_main_intermediate.yaml')

# #%%

# model = get_model(config)
# x = torch.zeros(10, 3, 512,512)
# y = model(x)
# # %%
# selected_output = y['depth_euclidean']  # Replace 'task1' with the actual key

# dot = make_dot(selected_output, params=dict(model.named_parameters()))
# dot.render('model_graph', format='png') 



# %%  (TEMPERORY) TO FIND THE MEAN ANDS STD DEVIATIONS OF THE PERFORMANCE ACROSS MULTIPLE TRIALS 

# import os
# import json
# import numpy as np
# import pandas as pd

# def read_json_file(file_path):
#     with open(file_path, 'r') as file:
#         return json.load(file)

# def process_data(data):
#     processed_data = {}

#     def process_key(key, values):
#         # Filter out non-numeric values
#         numeric_values = [v for v in values if isinstance(v, (int, float))]
        
#         # Skip if no numeric values
#         if not numeric_values:
#             return None

#         # Calculate mean and standard deviation
#         mean = np.mean(numeric_values)
#         std = np.std(numeric_values)
#         return {'mean': mean, 'std': std}

#     def extract_values(data, key):
#         return [item[key] for item in data if key in item and isinstance(item[key], (int, float, list))]

#     for key in data[0]:
#         if isinstance(data[0][key], dict):
#             # Process nested dictionary
#             for subkey, subvalue in data[0][key].items():
#                 sub_values = extract_values([d[key] for d in data if key in d], subkey)
#                 result = process_key(subkey, sub_values)
#                 if result:
#                     processed_data[f"{key}.{subkey}"] = result
#         else:
#             # Process normal key
#             values = extract_values(data, key)
#             result = process_key(key, values)
#             if result:
#                 processed_data[key] = result

#     return processed_data

# # List of directories
# # dir_name = '/proj/ltu_mtl/users/x_ricup/MTL_adaptive_results/inter/resnetd101/18_multi_depth_sn_res100_trial_'
# dir_name = '/proj/ltu_mtl/users/x_ricup/MTL_adaptive_results/meta/celebA/8_multi_seg_male_eyebrows_smile_lipstick_cheekbones_biglips_1e_5_trial_'
# num_trials = 3

# # Generate the directory names based on the given pattern
# directories = [f"{dir_name}{i}" for i in range(num_trials)]

# # Read all JSON files
# all_data = []
# for directory in directories:
#     for file_name in os.listdir(directory):
#         if file_name.endswith('.json'):
#             file_path = os.path.join(directory, file_name)
#             all_data.append(read_json_file(file_path))

# # Process the data
# result = process_data(all_data)

# # Print or use the result
# print(result)

# transformed_data = {key: f"{value['mean']:.4f} ± {value['std']:.4f}" for key, value in result.items()}
# # Convert the transformed data into a pandas DataFrame
# df = pd.DataFrame([transformed_data])

# # Specify the path for the Excel file in the last trial folder
# excel_file_path = f"{directories[-1]}/result.xlsx"

# # Save the DataFrame to an Excel file
# df.to_excel(excel_file_path, index=False)

# print(f"Results saved to {excel_file_path}")


# %%

