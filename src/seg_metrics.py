
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.module import Module
import numpy as np
import segmentation_models_pytorch as smp


class Seg_cross_entropy_metric(Module):
    def __init__(self, config):
        super(Seg_cross_entropy_metric, self).__init__()
        dataset_name = config['dataset_name']
        if dataset_name == 'Taskonomy':
            num_classes = 17
            self.wt_file = config['taskonomy_prior_factor']
            weight = torch.from_numpy(np.load(self.wt_file)).float().cuda()
        elif dataset_name == 'NYU':
            num_classes = config['seg_classes_NYU']
            weight = torch.ones(num_classes)*1.0
            weight[-1] = 0.1   #### other class 
            weight=weight.cuda()
        else:
            print('Not Implemented error')
        
        self.criterion = nn.CrossEntropyLoss(weight=weight)         

    def forward(self, out, label):
        with torch.no_grad():
            label = label.permute(0,2,3,1).reshape(-1,)
            num_class = out.shape[1]
            mask = label< num_class
            # mask = label !=0
            label = label[mask].int()
            logits = out.permute(0, 2, 3, 1).contiguous().view(-1,num_class)[mask]
            err = self.criterion(logits, label.long())            
            return err.cpu().numpy()
        
        

class calculate_IoU(Module):
    def __init__(self):
        super(calculate_IoU, self).__init__()

    def forward(self,gt, pred):
        with torch.no_grad():
            eps = 1e-8
            n_classes = pred.shape[1]            
            # pred = F.softmax(pred, dim=1)
            # pred = torch.argmax(pred, dim=1)
            iou = []
            for gt, pred in zip(gt, pred):
                tp, fp, fn = 0, 0, 0
                valid = (gt != 255)  #### ignore index 0 i.e. background
                for i_part in range(1,n_classes):
                    tmp_gt = (gt == i_part)
                    tmp_pred = (pred == i_part)
                    tp += torch.sum((tmp_gt & tmp_pred & valid).float())
                    fp += torch.sum((~tmp_gt & tmp_pred & valid).float())
                    fn += torch.sum((tmp_gt & ~tmp_pred & valid).float())
                jac = tp / max(tp + fp + fn, eps)
                iou.append(jac)
            iou = torch.mean(torch.tensor(iou))
            # iou = np.mean(iou)
            
            return iou.cpu().numpy()



            
            
            