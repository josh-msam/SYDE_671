import time
import torch
import tqdm
import numpy as np

from video import forward_video

def train_sequence(model,criterion,score_func,
                mode,images,masks,name,optimizer,batch_size,show=False):

    """
    Trains the sequence using batchees. It loads a sequence of
    batch_size into CPU memory. Then forward + back propegation
    is performed on the batch using the optimizer.
    
    Inputs:
        - dataloaders [dict of pytorch DataLoaders.]
        - model [pytorch model]
        - criterion [loss function]
        - optimizer 
        - batch_size [int number of frames in batch to consider]
        - mode [number of previous frames to maintain in memory]
        - num_epochs [number of epochs to use for training] 
    """
    
    start = 0                        
    jm = []
    m= None       
    num_images = images.shape[1] 
    video_loss = 0.0
    # LOOP OVER EVERY BATCH OF FRAMES IN VIDEO
    while(start < num_images ):

        if (num_images - start) >= batch_size:
            end = start+batch_size
        else:
            end = num_images

        # GET BATCH
        img_batch = images[:,start : end]
        msk_batch = masks[:,start : end]
        # FORWARD PASS
        jb,loss,mem = forward_video( model,criterion,score_func,mode,
                                    img_batch,msk_batch,name,mem=m, show=show) 
        # DO BACK PROP
        optimizer.zero_grad()    
        loss.backward()
        optimizer.step()                   

        video_loss += loss.item() 

        # DETACH MEMORY FROM COMPUTATIONAL GRAPH
        # SAVES MEMORY
        m = {k: v.detach() for k,v in mem.items()}
        del mem

        for j in jb: jm.append(j)
            
        start += batch_size                    
                 
    return jm, video_loss

                   
def train_model(dataloaders, model, criterion, optimizer, logger,score_func,
                batch_size=2,mode=0, num_epochs=25):

    """
    Trains the network. It loads an image sequence into CPU
    memory. Then a batch / sub-sequence from that video is taken
    and forward + back propegated through using the optimizer.
    
    The keys of the "dataloaders" dictionary should be:
        - "train" if backprop is wanted on that data
        - "val" if validation is wanted
 
    Mode determines the memory management settings:
        Mode = 0:     Use only the first ground truth frame information
        Mode = N > 0: Use first and N number of previous frames
        Mode = N < 0: [validation only] Use first and save every Nth frame
        
    Inputs:
        - dataloaders [dict of pytorch DataLoaders.]
        - model [pytorch model]
        - criterion [loss function]
        - optimizer 
        - batch_size [int number of frames in batch to consider]
        - mode [number of previous frames to maintain in memory]
        - num_epochs [number of epochs to use for training] 
    """
    
    d_str = '{}/{} {} L {:5f} C {:5f} J {:5f}'
    t_str = '\n\nTraining complete in {:.0f}m {:.0f}s \n\nBest Jaccard: {:4f}'  
      
    phases = list(dataloaders.keys())
    
    ### STARTING WITH THE TRAINING LOOP
    print("Start training \n")
    best_J = 0.0
    since = time.time()
    for e in range(num_epochs):

        # TRAIN AND EVALUATE EVERY EPOCH
        for phase in phases:
               
            torch.set_grad_enabled(phase == 'train')
                
            nr_img = 0
            e_loss = 0.0
            jaccard = []
            
            loader = dataloaders[phase]
            pbar = tqdm.tqdm(loader,position=0, leave=True)

            # LOOP OVER EVER VIDEO SEQUENCE 
            for images, masks, name in pbar:
  
                if phase == "val":
                    jm,loss,_ = forward_video(model,criterion,score_func,
                                              mode,images,masks,name)
                    video_loss = loss.item()
                    
                elif phase == "train" and mode <= 0:
                    jm,video_loss = train_sequence(model,criterion,score_func,
                                                mode,images,masks,name,
                                                optimizer,batch_size)
                # UPDATE STATISTICS
                e_loss += video_loss
                C = np.mean(jm)
                nr_img += images.shape[1]
                L = e_loss/nr_img
                jaccard.append(C)
                J = np.mean(jaccard)       
                sub = [e,num_epochs-1,phase,L,C,J]
                pbar.set_description(d_str.format(*sub))                              
                    
            # EPOCH FINISHED APPEND TO LOG FILE
            logger.add_row([e,phase,L,J])
            # IF BETTER VALIDATON SCORE --> SAVE NEW BEST MODEL
            if phase == 'val' and J > best_J:
                best_J = J       
                logger.save_model( model,[mode,e,J] )
        
    # FINISHED TRAINING
    time_elapsed = time.time() - since
    print(t_str.format( time_elapsed // 60, time_elapsed % 60, best_J))