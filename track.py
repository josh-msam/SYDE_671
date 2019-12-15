from datetime import datetime
import os
import csv
import numpy as np
import matplotlib.pyplot as plt
import torch

class Tracker:
    """
    THIS TRACKER IS USED TO SAVE IMPORTANT TRAINING INFORMATION. 
    
    """
    def __init__(self):
        self.time_str = "%m-%d-%Y_%H:%M:%S"
        
        self.date = None
        self.train_dir = None
        self.log_path = None
    
    def create_dir(self):    
        """
        Creates new tracking directory. 
        Creates new log file.
        """
        date = datetime.now().strftime(self.time_str)
        directory = "training_{}".format(date)
        wd = os.getcwd()
        self.train_dir = os.path.join(wd,directory)
        try:
            os.mkdir(self.train_dir)
        except OSError:
            raise("Creation of the directory {} failed".format(self.train_dir) )
            
        print("Training save directory: \n\n{}\n\n".format(self.train_dir) )
        log_file = "log_{}.csv".format(date)
        self.log_path = os.path.join(self.train_dir,log_file)
    
        with open( self.log_path, "w") as csv_file:
            writer = csv.writer(csv_file, delimiter=',')
            writer.writerow(["epoch","phase","loss","jaccard","time"])  

    def add_row(self, row):
        """
        Add a new row to the log file
        """
        t = datetime.now().strftime(self.time_str)
        row = row+[t]
        
        with open( self.log_path, "a+") as csv_file:
            writer = csv.writer( csv_file, delimiter=',')
            writer.writerow(row)
          
    def set_save_string(self, save_str):
        """ Set the string for model saving"""
        self.save_str = save_str
        
    def save_model(self, model, save_list ):
        """
        Save a pytorch model state_dict to disk
        
        """
        t = datetime.now().strftime(self.time_str)
        string = save_list+[t]
        state_file = self.save_str.format(*string) 
        state_path = os.path.join(self.train_dir,state_file)
        torch.save(model.state_dict(), state_path)
                
    def plot(self):
        """
        Plot the log file
        """
        with open( self.log_path, 'r') as csvfile:
            data = list(csv.reader(csvfile))
            
        data = np.array(data)
        
        loss = data[1:,2].astype(float)
        jaccard = data[1:,3].astype(float)
        
        s = data[1,1] == "val"
       
        v_loss = loss[not s::2]
        t_loss = loss[s::2]
        
        v_j = jaccard[not s::2]
        t_j = jaccard[s::2]
        
        fig, axs = plt.subplots(1, 2, constrained_layout=True)
        
        e = np.argmax(v_j)
        a = v_j[e]*100
        t_str = "[Run 4] {:3f}% validation accuracy at epoch {}".format(a,e)
        fig.suptitle(t_str, fontsize=10)
        
        axs[0].plot(v_loss,'r',label="Val-set loss")
        axs[0].plot(t_loss,'b',label="Train-set loss")
        
        axs[0].set_title('Cross entropy loss', fontsize=9)
        axs[0].set_xlabel('epoch', fontsize=8)
        axs[0].set_ylabel('loss', fontsize=8)
        axs[0].legend()
        axs[1].plot(v_j,'r',label="Val-set Jaccard")
        axs[1].plot(t_j,'b',label="Train-set Jaccard")
        
        axs[1].set_xlabel('epoch', fontsize=8)
        axs[1].set_title('Jaccard (Intersection over Union)', fontsize=9)
        axs[1].set_ylabel('jaccard', fontsize=8)
        
        axs[1].legend()
        
        plt.show()