import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from skimage.transform import resize

class Video:
    """
    VIDEO REPRESENTATION.
    
    Object stores all image paths of the video.
    
    The when called for frames it returns all frames
    in the video
    
    """
    
    def __init__(self, root, scale, name=None ):
        self.root = root
        self.name = name
        self.scale = scale
        self.paths = []

    def set_name( self, name ):
        self.name = name

    def get_name( self ):
        return self.name
    
    def add_path(self, img_path, msk_path ):
        ip = self.root+img_path
        mp =  self.root + msk_path 
        self.paths.append( [ip, mp] )
    
    def get_frames(self):
 
        images = []
        masks = []
        
        # LOOP OVER EVERY IMAGE PATH 
        for img_path, msk_path  in self.paths:
            im_frame = Image.open( img_path ).convert('RGB')
            image = np.array(im_frame) / 255.
            mk_frame = Image.open( msk_path ).convert('P')
            mask = np.array(mk_frame).astype(np.uint8)
          
            # RESCALE IMAGE 
            if self.scale < 1:
                r = int( image.shape[0] * self.scale )
                c = int( image.shape[1] * self.scale )
                image = resize(image, (r,c,3),preserve_range=True)
                mask = resize(mask, (r,c),preserve_range=True)
                
            images.append(image)
            masks.append(mask)

        i = np.array(images).astype(np.float32)
        m = np.array(masks).astype(np.uint8) / 255
        return i,m,self.name

    
class DAVIS(Dataset):
    """
    DAVIS 2016 DATASET FEED. 
    
    Saves all video and image paths. When indexed
    then it returns the video sequence.
    
    """

    def __init__(self, root, path, scale):

        self.videos = self.create_videos( root, path, scale )
        
    def __len__(self):
        return len(self.videos)
 
    def __getitem__(self, index):
        return self.videos[index].get_frames()
    
    def create_videos(self, root, path, scale ): 
        videos = []
        with open( root+path, "r") as lines:
            # create first video with unknown name
            current_video = Video( root ,scale )
            
            for line in lines:
                # line holds image and mask path
                img_path, msk_path = line.split()
                # video name is 3th folder
                video_name = img_path.split('/')[3]
                # get current video name for comparison
                current_name = current_video.get_name()
                # if no name then set name and add to video list
                if current_name == None:
                    current_video.set_name( video_name )
                    videos.append( current_video )
                # if name is different than make new video 
                # add to list and add image and mask paths
                elif current_name != video_name:
                    current_video = Video( root, scale, name=video_name )
                    videos.append( current_video )
                    current_video.add_path( img_path, msk_path )
                # otherwise then just add new image and mask path
                else:
                    current_video.add_path( img_path, msk_path )

        return videos


