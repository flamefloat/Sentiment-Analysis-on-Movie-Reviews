from GloVeEmb import glove_path
import pandas as pd
import torch 
print('start loading...')
glove = pd.read_csv( glove_path + '/glove.6B.100d.txt', sep='\ ', header = None, engine='python')
print('end loading...')
voca = glove[0].tolist()
voca_path = glove_path + '/voca.100d.pth'
torch.save(voca, voca_path)
print(voca[0:50], len(voca))