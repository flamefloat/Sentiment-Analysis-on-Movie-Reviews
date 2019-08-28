import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
#from torch.utils.data import DataLoader, TensorDataset
from GloVeEmb import test_data_tensor_path, test_Phrase_Len_Set_path, root_path, MODEL_PATH
from trainModel import SAmodel, DEVICE

test_sentiment_path = root_path + '/test_sentiment.pth'

def getSentiment(model, test_Phrase_data, test_Phrase_Len_Set):
    test_Phrase_data = test_Phrase_data.to(DEVICE)
    test_Phrase_Len_Set = test_Phrase_Len_Set.to(DEVICE)
    model.eval()
    with torch.no_grad():
        sentiment = model(test_Phrase_data, test_Phrase_Len_Set)
    return sentiment

if __name__ =='__main__':
    model_data = torch.load(MODEL_PATH)
    samodel = SAmodel().to(DEVICE)
    samodel.load_state_dict(model_data['net'])

    test_Phrase_data = torch.load(test_data_tensor_path)
    test_Phrase_Len_Set = torch.load(test_Phrase_Len_Set_path)
    sentiment = getSentiment(samodel, test_Phrase_data, test_Phrase_Len_Set)
    torch.save(sentiment, test_sentiment_path)
