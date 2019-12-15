import tqdm
import numpy as np
import torch

def get_loss( criterion, masks, Es ):
    """
    Return loss from criterion. Aranges dimensions
    """
    target = masks[0].type(torch.LongTensor)
    input = Es.permute(1,0,2,3)
    return criterion( input, target)
   
def get_score(func, masks, Es, ignore_first=False ):
    """
    Calculates score for every segmentation
    """
    input = Es.max(0)[1].type(torch.FloatTensor)
    target = masks[0].type(torch.LongTensor)
    I = input.detach().numpy() > 0
    T = target.detach().numpy() > 0
    rng = range(int(ignore_first),I.shape[0])
    return [func(I[i],T[i]) for i in rng]

def All_to_onehot( masks, K ):
    """
    To one hot encoding
    """
    M = torch.zeros((masks.shape[0], K, masks.shape[1], 
                      masks.shape[2], masks.shape[3]))
    
    for n in range(masks.shape[1]):
        for k in range( K ):
            M[:,k,n] = (masks[:,n] == k)
      
    return M

def forward_video(model,criterion,score_func,mode,images,masks,name,
                  mem=None,show=False):
    """
    Forward pass through sequence. 

    Mode determines the memory management settings:
        Mode = 0:     Use only the first ground truth frame information
        Mode = N > 0: Use first and N number of previous frames
        Mode = N < 0: [validation only] Use first and save every Nth frame
        
    Inputs:
        - model [pytorch model]
        - criterion [loss function]
        - score_func [function to quantify segmentation score]
        - mode [number of previous frames to maintain in memory]
        - name [used for printing to console]
        - mem [previous memory to use or not]
        - show [print output to console or not]
        
    Returns:
        - jm [Jaccard mean of the sequence]
        - loss [total loss of sequence]
        - mem [Updated memory]
    """
    Ms = All_to_onehot( masks, K=2)
    Fs = images.permute(0,4,1,2,3)                    
    
    if mode > 0:
        Es,mem = run_every(model, Fs, Ms, name, every=mode, mem=None,show=False)
    else:
        Es,mem = run_previ(model, Fs, Ms, name, nr=mode,mem=mem, show=False) 
    
    loss = get_loss(criterion, masks, Es[0])
    jm = get_score( score_func, masks, Es[0], ignore_first=True)
    return jm, loss,mem

def run_every( model, Fs, Ms, name, every=5,mem=None, show=False):
    """
    STM original implementation. Memorize with "every" intervals. 

    Inputs:
        - model [pytorch model]
        - Fs [query images ]
        - Ms [ground truth masks]
        - name [used for printing to console]
        - every [memorization interval]
        - mem [previous memory to use or not]
        - show [print output to console or not]
        
    Returns:
        - Es [segmentation probabilities]
    """
    
    num_frames = Fs.shape[2]
    to_memorize = [int(i) for i in np.arange(0, num_frames, step=every)]
    
    Es = torch.zeros_like(Ms)    
    Es[:,:,0] = Ms[:,:,0]
        
    pbar = range(int(mem==None), num_frames)
    if show: pbar = tqdm.tqdm(pbar,position=0, leave=True)
    
    for t in pbar:
        
        if show:
            pbar.set_description(name[0] )
        # memorize
        prev_key, prev_value = model(Fs[:,:,t-1], Es[:,:,t-1], 1) 

        if t-1 == 0: # 
            this_keys, this_values = prev_key, prev_value # only prev memory
        else:
            this_keys = torch.cat([keys, prev_key], dim=3)
            this_values = torch.cat([values, prev_value], dim=3)
        
        # segment
        logit = model(Fs[:,:,t], this_keys, this_values, 1)
        
        Es[:,:,t] = torch.nn.functional.softmax(logit, dim=1)
        
        # update
        if t-1 in to_memorize:
            keys, values = this_keys, this_values
        
    return Es, None


   
def run_previ( model, Fs, Ms, name, nr=0, mem=None, show=False):
    """
    NEW adaption. Memorize first + "nr" of previous frames.

    Inputs:
        - model [pytorch model]
        - Fs [query images ]
        - Ms [ground truth masks]
        - name [used for printing to console]
        - nr [number of previous frames to remember]
        - mem [previous memory to use or not]
        - show [print output to console or not]
        
    Returns:
        - Es [segmentation probabilities]
        - mem [Update memory]
    """
    
    Es = torch.zeros_like(Ms)                        
    num_frames = Fs.shape[2]

    args = [0]+list(range(nr,0))
    
    pbar = range(int(mem==None), num_frames)
    if show: 
        pbar = tqdm.tqdm(pbar,position=0, leave=True)
    
    if mem == None:
        # PUT FIRST FRAME INTO MEMORY
        Es[:,:,0] = Ms[:,:,0]  
        prev_key, prev_value = model( Fs[:,:,0], Es[:,:,0], 1) 
        mem = {'key': prev_key, 'val':prev_value}
       
    for t in pbar:      
        if show: 
            pbar.set_description( name[0] )

        # SEGMENT QUERY
        logit = model( Fs[:,:,t], mem['key'], mem['val'], 1)
       
        Es[:,:,t] = torch.nn.functional.softmax(logit, dim=1)
        
        # MEMORY SEGMENTATION
        prev_key, prev_value = model( Fs[:,:,t], Es[:,:,t], 1) 

        # APPEND TO MEMORY
        mem['key'] = torch.cat([mem['key'], prev_key], dim=3)
        mem['val'] = torch.cat([mem['val'], prev_value], dim=3)
        
        # REMOVE OLD MEMORY
        if mem['key'].shape[3] > (nr+1):
            mem['key'] = mem['key'][:,:,:,args]
            mem['val'] = mem['val'][:,:,:,args]

    return Es, mem
