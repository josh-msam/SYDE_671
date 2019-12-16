from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss

from dataset import DAVIS
from model import STM
from helpers import db_eval_iou, load_model, select_params
from track import Tracker
from train import train_model

if __name__ == "__main__":
    
    # main directory
    main_dir = "/home/gijs/Documents"

    # file access    
    p = {"/"      : main_dir+"/STM-master/DAVIS-data/DAVIS",
         "val"    :"/ImageSets/480p/val.txt",
         "train"  :"/ImageSets/480p/train.txt",
         "weights":"STM_weights.pth"}

    # weight containing hint will be trained
    weight_hint = ['Decoder'] 
    
    # settings
    s = 1                  # image scaling factor (1 is regular size)
    batch_size = 2         # number of images in the sequence for backprop
    mode = 0               # mode --> see train_model
    num_epochs = 10        # number of epochs for training
    learning_rate = 1e-5   # The learning rate
    new_arch = False       # Use adapted / changed Decoder architecture
    
    # Intersection over Union (Jaccard) used for scoring
    score_func = db_eval_iou
    
    # get the DAVIS 2016 data loaders
    loaders = {k: DataLoader(DAVIS(p["/"],p[k], s)) for k in ['train','val']}

    # get model and load pre-trained weights
    model = load_model( STM(new_arch=new_arch), p["weights"])

    # set trainable parameters
    select_params( model, contains=weight_hint)

    # loss function
    criterion = CrossEntropyLoss()
    
    # optimizier
    optimizer = Adam(model.parameters(), lr=learning_rate)

    # create logger
    log = Tracker()
    log.create_dir()
    log.set_save_string('state_dict_M_{}_E_{}_J_{:5f}_T_{}.pth' )
    
    # train model and validate after each epoch
    train_model(loaders, model, criterion, optimizer, log, score_func,
                batch_size=batch_size, mode=mode,num_epochs=num_epochs)

    # plot log file for statistics
    log.plot()
    
