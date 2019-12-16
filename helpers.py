import numpy as np
import torch


""" Compute Jaccard Index. """

def db_eval_iou(annotation,segmentation):

    """ Compute region similarity as the Jaccard Index.

    Arguments:
        annotation   (ndarray): binary annotation   map.
        segmentation (ndarray): binary segmentation map.

    Return:
        jaccard (float): region similarity

 """
# ----------------------------------------------------------------------------
# The 2017 DAVIS Challenge on Video Object Segmentation
#-----------------------------------------------------------------------------
# Copyright (c) 2017 Federico Perazzi
# Licensed under the BSD License [see LICENSE for details]
# Written by Federico Perazzi (federico@disneyresearch.com)
# Adapted from DAVIS 2016 (Federico Perazzi)
# ----------------------------------------------------------------------------
 
    annotation   = annotation.astype(np.bool)
    segmentation = segmentation.astype(np.bool)

    if np.isclose(np.sum(annotation),0) and np.isclose(np.sum(segmentation),0):
        return 1
    else:
        return np.sum((annotation & segmentation)) / \
                np.sum((annotation | segmentation),dtype=np.float32)
                
def load_model( model, weights_path ):
    """
    Load the model with pre-trained weights from STM paper.
    Puts model on the GPU and into evaluation mode such that
    the batchnormalization layers in the model are kept during
    training.
    
    Inputs:
        - model [pytorch model]
        - weights_path [filename of pre-trained weights state-dict]   
    """
    if torch.cuda.is_available():
        tm = torch.nn.DataParallel( model )
        tm.load_state_dict( torch.load( weights_path ), strict=False)  
        return tm.cuda().eval()
    else:
        raise("Cuda not available! ")
    
def select_params( model, contains=[]):
    """
    Freezes all weights and un-freezes particalur layers 
    which have the "hint" in their variable names.
    
    It prints which weights are unfrozen and how many
    parameters are trained.
    
    Inputs:
        - model [pytorch model]
        - contains [list of substring hints]   
    """
    names = []
    for name, param in model.named_parameters():
        param.requires_grad = False
        if any(name.find(c) != -1 for c in contains):
            param.requires_grad = True
            names.append([name,param.numel()])

    total = sum(p.numel() for p in model.parameters())       
    p_str = "\nParameters: \n\nTotal: {} \n\nTrained: {} \n"
    print(p_str.format(total,sum([p for n,p in names])))
    for n,t in names: print(n,t)
    print("\n")
    
def pad_divide_by(in_list, d, in_size):

    h, w = in_size
    
    if h % d > 0:
        new_h = h + d - h % d
    else:
        new_h = h
    if w % d > 0:
        new_w = w + d - w % d
    else:
        new_w = w
        
    lh, uh = int((new_h-h) / 2), int(new_h-h) - int((new_h-h) / 2)
    lw, uw = int((new_w-w) / 2), int(new_w-w) - int((new_w-w) / 2)
    
    pad_array = (int(lw), int(uw), int(lh), int(uh))
    out_list = [torch.nn.functional.pad(inp, pad_array) for inp in in_list]   

    return out_list, pad_array

def ToCuda(xs):
    if torch.cuda.is_available():
        if isinstance(xs, list) or isinstance(xs, tuple):
            return [x.cuda() for x in xs]
        else:
            return xs.cuda() 
    else:
        return xs
