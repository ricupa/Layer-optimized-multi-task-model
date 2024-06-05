#%%
import os
import gc
import torch
gc.collect()
import numpy as np
import sys
# import torch
import collections
from torchvision import datasets, models, transforms
from utils import *
import cv2
import matplotlib.pyplot as plt
import PIL
from PIL import Image
import pytorch_grad_cam
from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image

def get_class_names(config):
    class_labels = {0: 'background',
                    1: 'hair',
                    2: 'skin',
                    }
    return list(class_labels.values())

class SegmentationModelOutputWrapper(torch.nn.Module):
    
        def __init__(self, model): 
            super(SegmentationModelOutputWrapper, self).__init__()
            self.model = model
            
        def forward(self, x):
            return self.model(x)["segmentsemantic"]

class SemanticSegmentationTarget:
    def __init__(self, category, mask):
        self.category = category
        # self.mask = mask
        self.mask = torch.from_numpy(mask)
        # if torch.cuda.is_available():
        #     self.mask = self.mask.cuda()
        
    def __call__(self, model_output):
        return (model_output[self.category, :, : ] * self.mask).sum()

class MultiTaskModel_hooks(nn.Module):
    def __init__(self, backbone: nn.Module, decoders: nn.ModuleDict, tasks: list, config: dict):
        super(MultiTaskModel_hooks, self).__init__()
        # print(type(decoders.keys()), decoders.keys())
        # print(type(tasks), tasks)
        # assert(set(decoders.keys()) == set(tasks))
        self.backbone = backbone
        self.decoders = decoders
        self.tasks = tasks
        self.hook_outputs = {}

        def hook_fn(module, input, output, task_name):
            self.hook_outputs[task_name] = output

        # Register hooks
        self.hooks = []
        for task in tasks:
            temp_hook = config['inter_hooks'][task]
            layer = eval('self.backbone.'+ temp_hook)
            hook = layer.register_forward_hook(lambda module, input, output, t=task: hook_fn(module, input, output, t))
            self.hooks.append(hook)

    def forward(self, x):
        _ = self.backbone(x)
        task_outputs = {}
        for task in self.tasks:
            if task in self.hook_outputs:
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


def get_model_hooks(config):
    
    backbone, bb_channels = get_backbone(config, config['num_input_ch'])           
    heads = torch.nn.ModuleDict({task: get_head(config, backbone, task) for task in config['task_list']})         
    model = MultiTaskModel_hooks(backbone, heads, config['task_list'], config)           
    
        # from utils import get_model
        # model = get_model(config)
        # model = MultiTaskModel(backbone, heads, config['task_list'])
    return model 
#%%

dir = '8_multi_seg_male_eyebrows_smile_lipstick_cheekbones_biglips_nosparse_trial_0/'
# name = "/proj/ltu_mtl/users/x_ricup/MTL_adaptive_results/inter/celebA/resnetd50/" + dir

name = "/proj/ltu_mtl/users/x_ricup/MTL_adaptive_results/meta/celebA/" + dir

config_exp = name+ 'config_file.yaml'
config = create_config(config_exp)

out_folder = 'grad_cam/'+ config['backbone']  +'/'

fname = out_folder + dir 
if not os.path.exists(fname):
    os.makedirs(fname)
else:
    print('folder already exist')
#%%  
################################################################
# ##LOAD IMAGE
device = 'cpu'
imgname = "/proj/ltu_mtl/dataset/celebA/CelebA-HQ-img/5553.jpg"    #### 0.jpg to 28000.jpg
seg_mask_path = '/proj/ltu_mtl/dataset/celebA/CelebA-HQ-segmask/05553_mask.npy'
img = PIL.Image.open(imgname)
img_rgb = img.resize((256,256))  ###, PIL.Image.ANTIALIAS
img_rgb = np.array(img_rgb)
img_rgb = np.float32(img_rgb) / 255
transform_v = [transforms.Resize((256,256)), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
img_transform = transforms.Compose(transform_v) 
img_2 = img_transform(img)
img2 = img_2.unsqueeze(0)
img2 = img2.to(device)
plt.imshow(img_rgb)
plt.axis('off')
#%%

seg_mask = mask = np.load(seg_mask_path)
resized_mask = np.resize(mask, (32, 32))
plt.imshow(seg_mask)
plt.axis('off')
# plt.savefig(fname + 'gt_mask1.jpg', bbox_inches='tight', pad_inches=0)

#%%

# img_rgb1 = img.resize((32,32)) ### k=just for the sake of the paper 
# img.save(fname+'img3.jpg')

# %% Load model checkpoint 

checkpoint = torch.load(name + '/checkpoint.pt', map_location= device)
# checkpoint = torch.load(dir_checkpoint + '/checkpoint.pt', map_location= device)    #### segmentsemantic_

if 'inter_hooks' in config.keys():
    model1 = get_model_hooks(config)
else:
    from utils import get_model
    model1= get_model(config)
# model1 = get_model(config)
model1 = model1.to(device)   
if (config['setup'] == 'multitask') and (config['comb_loss'] == 'uncertainity'):                   
    log_vars = {task: nn.Parameter(torch.tensor(1.0, device = device, requires_grad = True,dtype=torch.float64))  for task in config['task_list']}           
    
    for task , value in log_vars.items():
        model1.register_parameter(task, value)
                
         
checkpoint = torch.load(name + '/checkpoint.pt', map_location=device)
print('load the checkpoint')
model1.load_state_dict(checkpoint['model'], strict = False) 
model1.eval()
backbone = model1.backbone
Decoder = model1.decoders
sem_classes = get_class_names(config)
sem_class_to_idx = {cls: idx for (idx, cls) in enumerate(sem_classes)}
input_tensor = preprocess_image(img_rgb,mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])  
model = SegmentationModelOutputWrapper(model1)
output = model(img2)
# print(output.shape)
# output = F.interpolate(output, size=(256,256), mode="bilinear")
normalized_masks = torch.nn.functional.softmax(output, dim=1).cpu()
normalized_masks = normalized_masks.argmax(axis=1)
mask = normalized_masks[0, :, :].detach().cpu().numpy()


# %%

output1 = F.interpolate(output, size=(256,256), mode="bilinear")
normalized_masks1 = torch.nn.functional.softmax(output1, dim=1).cpu()
normalized_masks1 = normalized_masks1.argmax(axis=1)
mask1 = normalized_masks1[0, :, :].detach().cpu().numpy()
plt.axis('off')
plt.imshow(mask1)
# plt.savefig(fname+'pred_mask1.jpg', bbox_inches='tight', pad_inches=0)
# %%
##########################################################################
category = sem_class_to_idx["skin"]

mask_float = np.float32(mask == category)

targets = [SemanticSegmentationTarget(category, mask_float)]

# target_layers =  [Decoder.segmentsemantic[0].convs[3][0]]

target_layers =  [backbone.layer4[-2].conv2]

with GradCAM(model=model, target_layers=target_layers, use_cuda=False) as cam:
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0, :]
    cam_image = show_cam_on_image(img_rgb, grayscale_cam, use_rgb=True)

Image.fromarray(cam_image)

#%%
pil_img = Image.fromarray(cam_image)
# Save the image
output_path = fname+'cam_skin1.jpg'  # Specify the path and filename
pil_img.save(output_path)

#%%
# for name, param in model1.named_parameters():
#     print(name)

# file = "/proj/ltu_mtl/dataset/celebA/CelebAMask-HQ-attribute-anno.txt"

# df = pd.read_csv(file, sep=' ')

# # Save the DataFrame to an Excel file
# # Replace 'output.xlsx' with your desired Excel file name
# df.to_excel('celebA_attr.xlsx')
# %%
####### grad cam for classification 

def get_output_for_task(input_tensor, task):
    # Forward pass through the backbone
    backbone_output = model['backbone'](input_tensor)
    # Forward pass through the specific task head
    task_output = model[task](backbone_output)
    return task_output

def forward_task_specific(model, input_tensor, task):
    model_output = model(input_tensor)
    return model_output[task]

class TaskSpecificModel(nn.Module):
    def __init__(self, multitask_model, task_name):
        super(TaskSpecificModel, self).__init__()
        self.multitask_model = multitask_model
        self.task_name = task_name

    def forward(self, x):
        # Assuming the multitask model returns a dict of {task_name: task_output}
        task_outputs = self.multitask_model(x)
        # Return the output tensor for the specified task
        return task_outputs[self.task_name]
    
    
class BinaryClassificationTarget:
    def __init__(self, threshold=0.5):
        self.threshold = threshold

    def __call__(self, model_output):
        # Assume model_output is a tensor with shape [batch_size] and contains probabilities of the positive class
        # This example uses a threshold to decide the class, but you could adjust the logic as needed      
        if model_output >= self.threshold:
            return model_output  # Use the model's output directly as the target for Grad-CAM
        else:
            return 1 - model_output 
        
#%%
##################################
# Example for a single task; repeat as needed for other tasks
task_name = 'class_male'  # Specify the task
target_layers = [backbone.layer3[5].conv1]
# Wrap your multi-task model for the specific task
temp_model = TaskSpecificModel(model1, task_name)

# Assuming 'target_layers' is correctly defined as per your model's architecture
cam = GradCAM(model=temp_model, target_layers=target_layers, use_cuda=False)
targets = [BinaryClassificationTarget(threshold=0.3)]

# Now, use Grad-CAM as usual, focusing on the positive class (class index 1 for binary classification)
grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0,:]
if grayscale_cam.ndim == 3 and grayscale_cam.shape[0] == 1:
    grayscale_cam = grayscale_cam.squeeze(0) 
    
# grayscale_cam = np.uint8(255 * grayscale_cam)

# Apply the colormap to generate the heatmap
# heatmap = cv2.applyColorMap(grayscale_cam, cv2.COLORMAP_JET)
cam_image = show_cam_on_image(img_rgb, grayscale_cam, use_rgb=True)
Image.fromarray(cam_image)


# %%
pil_img = Image.fromarray(cam_image)
# Save the image
output_path = fname+'class_male_1.jpg'  # Specify the path and filename
pil_img.save(output_path)
# %% ###########################################################################
# Example for a single task; repeat as needed for other tasks
task_name = 'class_smile'  # Specify the task
target_layers = [backbone.layer3[5].conv2]
# Wrap your multi-task model for the specific task
temp_model = TaskSpecificModel(model1, task_name)

# Assuming 'target_layers' is correctly defined as per your model's architecture
cam = GradCAM(model=temp_model, target_layers=target_layers, use_cuda=False)
targets = [BinaryClassificationTarget(threshold=0.3)]

# Now, use Grad-CAM as usual, focusing on the positive class (class index 1 for binary classification)
grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0,:]
if grayscale_cam.ndim == 3 and grayscale_cam.shape[0] == 1:
    grayscale_cam = grayscale_cam.squeeze(0) 

cam_image = show_cam_on_image(img_rgb, grayscale_cam, use_rgb=True)
Image.fromarray(cam_image)


# %%
pil_img = Image.fromarray(cam_image)
# Save the image
output_path = fname+'class_smile_1.jpg'  # Specify the path and filename
pil_img.save(output_path)
# %% ###########################################################################
# Example for a single task; repeat as needed for other tasks
task_name = 'class_highcheekbones'  # Specify the task
target_layers = [backbone.layer3[4].conv2]
# Wrap your multi-task model for the specific task
temp_model = TaskSpecificModel(model1, task_name)

# Assuming 'target_layers' is correctly defined as per your model's architecture
cam = GradCAM(model=temp_model, target_layers=target_layers, use_cuda=False)
targets = [BinaryClassificationTarget(threshold=0.3)]

# Now, use Grad-CAM as usual, focusing on the positive class (class index 1 for binary classification)
grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0,:]
if grayscale_cam.ndim == 3 and grayscale_cam.shape[0] == 1:
    grayscale_cam = grayscale_cam.squeeze(0) 
    
# grayscale_cam = np.uint8(255 * grayscale_cam)

# Apply the colormap to generate the heatmap
# heatmap = cv2.applyColorMap(grayscale_cam, cv2.COLORMAP_JET)
cam_image = show_cam_on_image(img_rgb, grayscale_cam, use_rgb=True)
Image.fromarray(cam_image)

# %%
pil_img = Image.fromarray(cam_image)
# Save the image
output_path = fname+'class_CB_1.jpg'  # Specify the path and filename
pil_img.save(output_path)


# %% ###########################################################################
# Example for a single task; repeat as needed for other tasks
task_name = 'class_lipstick'  # Specify the task
target_layers = [backbone.layer3[5].conv1]
# Wrap your multi-task model for the specific task
temp_model = TaskSpecificModel(model1, task_name)

# Assuming 'target_layers' is correctly defined as per your model's architecture
cam = GradCAM(model=temp_model, target_layers=target_layers, use_cuda=False)
targets = [BinaryClassificationTarget(threshold=0.3)]

# Now, use Grad-CAM as usual, focusing on the positive class (class index 1 for binary classification)
grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0,:]
if grayscale_cam.ndim == 3 and grayscale_cam.shape[0] == 1:
    grayscale_cam = grayscale_cam.squeeze(0) 
    
# grayscale_cam = np.uint8(255 * grayscale_cam)

# Apply the colormap to generate the heatmap
# heatmap = cv2.applyColorMap(grayscale_cam, cv2.COLORMAP_JET)
cam_image = show_cam_on_image(img_rgb, grayscale_cam, use_rgb=True)
Image.fromarray(cam_image)
# %%
pil_img = Image.fromarray(cam_image)
# Save the image
output_path = fname+'class_lipstick_1.jpg'  # Specify the path and filename
pil_img.save(output_path)

# %% ###########################################################################

# Example for a single task; repeat as needed for other tasks
task_name = 'class_eyebrows'  # Specify the task
target_layers = [backbone.layer3[3].conv2]
# Wrap your multi-task model for the specific task
temp_model = TaskSpecificModel(model1, task_name)

# Assuming 'target_layers' is correctly defined as per your model's architecture
cam = GradCAM(model=temp_model, target_layers=target_layers, use_cuda=False)
targets = [BinaryClassificationTarget(threshold=0.3)]

# Now, use Grad-CAM as usual, focusing on the positive class (class index 1 for binary classification)
grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0,:]
if grayscale_cam.ndim == 3 and grayscale_cam.shape[0] == 1:
    grayscale_cam = grayscale_cam.squeeze(0) 
    
# grayscale_cam = np.uint8(255 * grayscale_cam)

# Apply the colormap to generate the heatmap
# heatmap = cv2.applyColorMap(grayscale_cam, cv2.COLORMAP_JET)
cam_image = show_cam_on_image(img_rgb, grayscale_cam, use_rgb=True)
Image.fromarray(cam_image)
# %%
pil_img = Image.fromarray(cam_image)
# Save the image
output_path = fname+'class_eyebrow_3.jpg'  # Specify the path and filename
pil_img.save(output_path)


# %% ###########################################################################

# Example for a single task; repeat as needed for other tasks
task_name = 'class_biglips'  # Specify the task
target_layers = [backbone.layer4[0].conv1]
# Wrap your multi-task model for the specific task
temp_model = TaskSpecificModel(model1, task_name)

# Assuming 'target_layers' is correctly defined as per your model's architecture
cam = GradCAM(model=temp_model, target_layers=target_layers, use_cuda=False)
targets = [BinaryClassificationTarget(threshold=0.3)]

# Now, use Grad-CAM as usual, focusing on the positive class (class index 1 for binary classification)
grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0,:]
if grayscale_cam.ndim == 3 and grayscale_cam.shape[0] == 1:
    grayscale_cam = grayscale_cam.squeeze(0) 
    
# grayscale_cam = np.uint8(255 * grayscale_cam)

# Apply the colormap to generate the heatmap
# heatmap = cv2.applyColorMap(grayscale_cam, cv2.COLORMAP_JET)
cam_image = show_cam_on_image(img_rgb, grayscale_cam, use_rgb=True)
Image.fromarray(cam_image)
# %%
pil_img = Image.fromarray(cam_image)
# Save the image
output_path = fname+'class_biglips_1.jpg'  # Specify the path and filename
pil_img.save(output_path)
# %%
