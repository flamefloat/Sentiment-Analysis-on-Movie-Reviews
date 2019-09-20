"""
采用GolVe预训练词向量对输入的词序列编码
"""
import numpy as np
import torch
import pandas as pd
import time
root_path = 'G:/MHwork/MHdata/NLP/Sentiment-Analysis-on-Movie-Reviews'
glove_path = 'G:/MHwork/MHdata/NLP/glove'
voca_path = glove_path + '/voca.100d.pth'
# training data
train_data_tensor_path = root_path + '/embdata/train_data_tensor.pth'
Phrase_Len_Set_path = root_path + '/embdata/Phrase_Len_Set.pth'
Sentiment_Set_path = root_path + '/embdata/Sentiment_Set.pth'
#test data
test_data_tensor_path = root_path + '/embdata/test_data_tensor.pth'
test_Phrase_Len_Set_path = root_path + '/embdata/test_Phrase_Len_Set.pth'


def token2vector(data, glove, voca, isTrian_data = True):
    #UNK = torch.zeros(100) # unknown word, OOV
    Phrase_Set = data['Phrase']
    Phrase_Len_Set = []
    Sentiment_Set = []
    batch = len(Phrase_Set)
    for i in range(batch):
        Phrase_Len_Set.append(len(Phrase_Set[i].split(' ')))
    max_len = max(Phrase_Len_Set)
    data_tensor = torch.zeros(batch, max_len, 100)
    for i in range(batch): # batch
        list_Phrase_Set = Phrase_Set[i].split(' ')
        for j in range(Phrase_Len_Set[i]): # seq_len
            #print('*********', list_Phrase_Set[j])
            if list_Phrase_Set[j] in voca:
                data_tensor[i,j,:] = torch.tensor(glove.loc[list_Phrase_Set[j]].tolist()) # dim = 100
            #elif list_Phrase_Set[j].title() in voca:
            #    data_tensor[i,j,:] = torch.tensor(glove.loc[list_Phrase_Set[j.title()]].tolist()) #Title
            #elif list_Phrase_Set[j].lower() in voca:
            #    data_tensor[i,j,:] = torch.tensor(glove.loc[list_Phrase_Set[j.lower()]].tolist()) #lower
            #elif list_Phrase_Set[j].upper() in voca:
            #    data_tensor[i,j,:] = torch.tensor(glove.loc[list_Phrase_Set[j.upper()]].tolist()) #UPPER
            else:
                pass
                #data_tensor[i,j,:] = UNK
    Phrase_Len_Set = torch.tensor(Phrase_Len_Set)
    if isTrian_data:
        Sentiment_Set = data['Sentiment']
        Sentiment_Set = torch.tensor(Sentiment_Set)
    return data_tensor, Phrase_Len_Set, Sentiment_Set #[batch, max_len, dim], [bath], [bath]

if __name__ == '__main__':
    time_start = time.time()
    train_data = pd.read_csv( root_path + '/data/train.tsv',sep='\t')
    test_data = pd.read_csv( root_path + '/data/test.tsv',sep='\t')
    glove = pd.read_csv( glove_path + '/glove.6B.100d.txt', sep='\ ', header = None, index_col = 0,  engine='python')
    #a = 'the'
    #aa = torch.tensor(glove.loc[a.title()].tolist())
    #print('++++++',aa)
    voca = torch.load(voca_path)
    time_end = time.time()
    print('load time cost:%fs'%(time_end-time_start))

    print('start embedding...')
    time_start = time.time()
    # train_data embedding
    train_data_tensor, Phrase_Len_Set, Sentiment_Set = token2vector(train_data, glove, voca, isTrian_data = True)
    torch.save(train_data_tensor, train_data_tensor_path)
    torch.save(Phrase_Len_Set, Phrase_Len_Set_path)
    torch.save(Sentiment_Set, Sentiment_Set_path)

    time_end1 = time.time()
    print('train data embedding time cost:%fs'%(time_end1-time_start))
    # test_data embedding
    test_data_tensor, test_Phrase_Len_Set, Sentiment_Set = token2vector(test_data, glove, voca, isTrian_data = False)
    torch.save(test_data_tensor, test_data_tensor_path)
    torch.save(test_Phrase_Len_Set, test_Phrase_Len_Set_path)
    print('end embedding.')

    time_end2 = time.time()
    print('test data embedding time cost:%fs'%(time_end2-time_end1))

    
