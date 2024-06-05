

# def channelwise_groups_fn(params):
#     return [param.view(param.shape[1], -1).transpose(1, 0) for param in params]

def column_groups_fn(params):
    return [param.unsqueeze(0) for param in params]

def conv_groups_fn(params):
    return [param.view(param.shape[:2] + (-1,)).transpose(2, 1) for param in params]

def resnet_groups(model, args):
    to_prox_conv, to_prox_linear = [], []
    remaining = []

    for name, param in model.named_parameters():
        if 'backbone' not in name:
            remaining.append(param)
            continue
        if 'weight' not in name:
            remaining.append(param)
            continue
        if param.ndim != 4:
            remaining.append(param)
            continue
        to_prox_conv.append(param) ### accumulates the convolutional weights from the ResNet's backbone which you wish to process with a group-wise operation

    optimizer_grouped_parameters = [             
        {
            'params': to_prox_conv,  
            'weight_decay': args['weight_decay'],
            'groups_fn': conv_groups_fn  #### conv_groups_fn
        }
    ]
    if remaining:    #########Holds all other parameters (like biases, and possibly BatchNorm parameters) that aren't convolutional weights from the backbone.
        optimizer_grouped_parameters.append({
            'params': remaining,
            'weight_decay': args['weight_decay'],
            'groups_fn': None
        })

    return optimizer_grouped_parameters




# def resnet_groups(model, args):
#     to_prox_conv, to_prox_linear = [], []
#     remaining = []

#     for name, param in model.named_parameters():
#         if 'weight' in name:
#             if param.ndim == 4:
#                 to_prox_conv.append(param)
#             elif param.ndim == 2:
#                 to_prox_linear.append(param)
#             else:
#                 remaining.append(param)  # BN weight
#         else:
#             remaining.append(param)

#     optimizer_grouped_parameters = [
#         {
#             'params': to_prox_conv,
#             'weight_decay': args.weight_decay,
#             'groups_fn': conv_groups_fn
#         },
#         {
#             'params': to_prox_linear,
#             'weight_decay': args.weight_decay,
#             'groups_fn': column_groups_fn
#         }
#     ]
#     if remaining:
#         optimizer_grouped_parameters.append({
#             'params': remaining,
#             'weight_decay': args.weight_decay,
#             'groups_fn': None
#         })

#     return optimizer_grouped_parameters