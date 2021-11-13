import os 
import json 
import torch
from slimnet import SlimNet
from torchvision import transforms
from PIL import Image
import numpy as np
from glob import glob 
from tqdm import tqdm 
from multiprocessing import Pool 
import torch.multiprocessing as mp


DEST_PATH = '/raid/t-yazen/datasets/ravdess_attributes'
PATH_TO_IMAGE = '/raid/t-yazen/datasets/ravdess_vllp/'
os.makedirs(DEST_PATH, exist_ok=True)

labels = np.array(['5_o_Clock_Shadow', 'Arched_Eyebrows', 'Attractive', 'Bags_Under_Eyes',
       'Bald', 'Bangs', 'Big_Lips', 'Big_Nose', 'Black_Hair', 'Blond_Hair',
       'Blurry', 'Brown_Hair', 'Bushy_Eyebrows', 'Chubby', 'Double_Chin',
       'Eyeglasses', 'Goatee', 'Gray_Hair', 'Heavy_Makeup', 'High_Cheekbones',
       'Male', 'Mouth_Slightly_Open', 'Mustache', 'Narrow_Eyes', 'No_Beard',
       'Oval_Face', 'Pale_Skin', 'Pointy_Nose', 'Receding_Hairline',
       'Rosy_Cheeks', 'Sideburns', 'Smiling', 'Straight_Hair', 'Wavy_Hair',
       'Wearing_Earrings', 'Wearing_Hat', 'Wearing_Lipstick',
       'Wearing_Necklace', 'Wearing_Necktie', 'Young'])
# GPU isn't necessary but could definitly speed up, swap the comments to use best hardware available
#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device('cuda')
transform = transforms.Compose([
                              transforms.Resize((178,218)),
                              transforms.ToTensor(),
                              transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ])

video_list = sorted([os.path.join(PATH_TO_IMAGE, i) for i in os.listdir(PATH_TO_IMAGE) if '.' not in i])[500:]

def worker(gpuid): 
    torch.cuda.set_device(gpuid)
    model = SlimNet.load_pretrained('models/celeba_20.pth').cuda()
    model.eval() 
    
    for i in tqdm(range(gpuid, len(video_list), 4)):
        vpath = video_list[i]
        frame_list = sorted(list(glob(os.path.join(vpath, '*.png'))))
        if len(frame_list) == 0: 
            return  
        img = []
        # load image to tensor 
        for frame in frame_list: 
            with open(frame, 'rb') as f:
                x = transform(Image.open(f)).unsqueeze(0).to(device)
            img.append(x)
        x = torch.cat(img, dim=0)

        # inference 
        with torch.no_grad():
            logits = model(x)
            sigmoid_logits = torch.sigmoid(logits).squeeze().cpu().numpy()

        # save results
        info = {
            'video_name': os.path.basename(vpath), 
            'frames_logits': []}
        for i in range(len(frame_list)): 
            info['frames_logits'].append({os.path.basename(frame_list[i]): sigmoid_logits[i].tolist()})
        video_logits = sigmoid_logits.mean(0)
        info['video_attribute'] = labels[(video_logits>0.45).astype(bool)].tolist()
        info['video_logits'] = video_logits.tolist()

        with open(os.path.join(DEST_PATH, os.path.basename(vpath)+'.json'), 'w') as f:
            json.dump(info, f)


if __name__ == '__main__': 
    
    # Make tensor and normalize, add pseudo batch dimension and move to configured device 
    mp.spawn(worker, nprocs=4)