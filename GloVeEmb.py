"""
采用GolVe预训练词向量对输入的词序列编码
"""
import numpy as np
import torch
import pandas as pd
train_data = pd.read_csv('C:/Users/MH/Desktop/MyCode/NLP/Sentiment-Analysis-on-Movie-Reviews/data/train.tsv',sep='\t')
glove = pd.read_csv('C:/Users/MH/Desktop/MyCode/NLP/glove.840B.300d/go.txt', sep=' ', header = None, index_col = 0)
root_path = 'C:/Users/MH/Desktop/MyCode/NLP/Sentiment-Analysis-on-Movie-Reviews'
train_data_tensor_path = root_path + '/train_data_tensor.pth'
Phrase_Len_Set_path = root_path + '/Phrase_Len_Set.pth'
Sentiment_Set_path = root_path + '/Sentiment_Set.pth'

def token2vector(train_data, glove):
    Phrase_Set = train_data['Phrase']
    Phrase_Len_Set = []
    Sentiment_Set = train_data['Sentiment']
    batch = len(Sentiment_Set)
    for i in range(batch):
        Phrase_Len_Set.append(len(Phrase_Set[i].split(' ')))
    max_len = max(Phrase_Len_Set)
    train_data_tensor = torhc.zeros(batch, max_len, 300)
    for i in range(batch): # batch
        list_Phrase_Set = Phrase_Set[i].split(' ')
        for j in range(Phrase_Len_Set[i]): # seq_len
            train_data_tensor[i,j,:] = torch.tensor(glove.loc[list_Phrase_Set[j]].tolist()) # dim = 300
    torch
    Phrase_Len_Set = torch.tensor(Phrase_Len_Set)
    Sentiment_Set = torch.tensor(Sentiment_Set)
    return train_data_tensor, Phrase_Len_Set, Sentiment_Set #[batch, max_len, dim], [bath], [bath]

if __name__ == '__main__':
    print('start embedding...')
    train_data_tensor, Phrase_Len_Set, Sentiment_Set = token2vector(train_data, glove)
    torch.save(train_data_tensor, train_data_tensor_path)
    torch.save(Phrase_Len_Set, Phrase_Len_Set_path)
    torch.save(Sentiment_Set, Sentiment_Set_path)
    print('end embedding.')

    
