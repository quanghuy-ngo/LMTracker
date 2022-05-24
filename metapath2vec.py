# Reaches around 91.8% Micro-F1 after 5 epochs.

from asyncio import FastChildWatcher
from curses import meta
from locale import normalize
import os.path as osp
from typing import final
from xxlimited import new
import os, datetime
import torch
import time
import os, datetime
from torch_geometric.datasets import AMiner
from tqdm import tqdm
# import torch.multiprocessing
# torch.multiprocessing.set_sharing_strategy('file_system')
from model import MetaPath2Vec # modified Metapath2vec for the path embedding generation
import math

debug_mode = True
epoch = 100
dir_save = '/home/andrewngo/Desktop/MLTracker/graph_data_20220305160351'
data = torch.load(dir_save + '/graph_data_torch.pt')
print(data)

metapath_strat = "CUC"

# C -> U -> C -> U -> C
if metapath_strat == "CUC":
    metapath = [
        ('Computer', 'Logon_rev', 'User'),
        ('User', 'Logon', 'Computer'),
        ('Computer', 'Logon_rev', 'User'),
        ('User', 'Logon', 'Computer')
    ]
elif metapath_strat == "UCAC":
# U -> C -> A -> C -> A -> C -> U
    metapath = [
        ('User', 'Logon', 'Computer'),
        ('Computer', 'Use', 'Auth_Type'),
        ('Auth_Type', 'Use_rev','Computer'),
        ('Computer', 'Use', 'Auth_Type'),
        ('Auth_Type', 'Use_rev','Computer'),
        ('Computer', 'Logon_rev', 'User')
    ]
elif metapath_strat == "UCCA":
    metapath = [
        ('User', 'Logon', 'Computer'),
        ('Computer', 'Connect', 'Computer'),
        ('Computer', 'Use','Auth_Type'),
        ('Auth_Type', 'Use_rev', 'Computer'),
        ('Computer', 'Connect','Computer'),
        ('Computer', 'Logon_rev', 'User')
    ]
elif metapath_strat == "UCC":
    metapath = [
        ('User', 'Logon', 'Computer'),
        ('Computer', 'Connect', 'Computer'),
        ('Computer', 'Connect', 'Computer'),
        ('Computer', 'Logon_rev', 'User')
    ]







# metapath = [
#     ('Computer', 'Logon_rev', 'User'),
#     ('User', 'Logon', 'Computer'),
#     ('Computer', 'Create', 'Process'),
#     ('Process', 'Create_rev', 'Computer'),
#     ('Computer', 'Logon_rev', 'User'),
#     ('User', 'Logon', 'Computer'),
# ]

# metapath = [
#     ('Computer', 'Logon_rev', 'User'),
#     ('User', 'Logon', 'Computer'),
#     ('Computer', 'Create', 'Process'),
#     ('Process', 'Create_rev', 'Computer'),
#     ('Computer', 'Logon_rev', 'User'),
#     ('User', 'Logon', 'Computer'),
# ]


# metapath = [
#     ('Computer', 'Logon_rev', 'User'),
#     ('User', 'Logon', 'Computer'),
#     ('Computer', 'Create', 'Process'),
#     ('Process', 'Create_rev', 'Computer'),
#     ('Computer', 'Logon_rev', 'User'),
#     ('User', 'Logon', 'Computer'),
# ]



device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = MetaPath2Vec(data.edge_index_dict, embedding_dim=128,
                    metapath=metapath, walk_length=100, context_size=10,
                    walks_per_node=8, num_negative_samples=5,
                    sparse=True).to(device)

loader = model.loader(batch_size=32, shuffle=True, num_workers=6)
optimizer = torch.optim.SparseAdam(list(model.parameters()), lr=0.005)



# generate metapath



def generate_metapath_sampling(sample=20000):
    node_num = len(metapath) + 1
    model_sample = MetaPath2Vec(data.edge_index_dict, embedding_dim=128,
                        metapath=metapath, walk_length=node_num, context_size=node_num,
                        walks_per_node=1, num_negative_samples=1,
                        sparse=True).to(device)
    loader_sample = model_sample.loader(batch_size=1, shuffle=False, num_workers=1)
    optimizer = torch.optim.SparseAdam(list(model_sample.parameters()), lr=0.001)
    print(model_sample.start)
    print(model_sample.end)
    model_sample.train()
    temp = []
    for i, (pos_rw, neg_rw) in enumerate(tqdm(loader_sample)):
        temp.append(pos_rw.to(device)[0])
        if i > sample:
            break
    path = temp

    new_path = []
    print(path[1])
    for i in range(len(path)):
        if int(list(path[i])[-1]) != int(list(path[i])[-2]):
            new_path.append(path[i])
    # print(len(new_path))
    path = new_path


    temp = []
    for i in range(len(path)):


        if len(metapath)%2 == 0:
            temp.append(path[i][:math.ceil((len(metapath)+1)/2)])
            temp.append(path[i][math.ceil((len(metapath)+1)/2)-1:])
        else:
            temp.append(path[i][:math.ceil((len(metapath)+1)/2)])
            temp.append(path[i][math.ceil((len(metapath)+1)/2):])
    path = temp
    path = list(set(path))
    # print(path)
    print(len(path))
    return temp, model_sample.start, model_sample.end



# remember re-implement this function when 

def reindexing_path(model, malicious_path, path_type):
# ''' malicious_path of each node type are arranging from 0
# however, in metapath2vec implementation, they are convert to 
# a unique indexing (i.e, computer from 0 -> 32131, user from 32131 -> 42213). 
# This function will convert to metapath2vec reference indexing. '''
    new_path = []
    if path_type == "CUC":
        for i in malicious_path:
            first = i[0] + model.start["Computer"]
            sec = i[1] + model.start["User"]
            third = i[2] + model.start["Computer"]
            new_path.append((first, sec, third))
        # print(malicious_path)

    elif path_type == "UCAC":
        for i in malicious_path:
            first = i[0] + model.start["User"]
            sec = i[1] + model.start["Computer"]
            third = i[2] + model.start["Auth_Type"]
            forth = i[3] + model.start["Computer"]
            new_path.append((first, sec, third, forth))
    elif path_type == "UCCA":
        for i in malicious_path:
            first = i[0] + model.start["User"]
            sec = i[1] + model.start["Computer"]
            third = i[2] + model.start["Computer"]
            forth = i[3] + model.start["Auth_Type"]
            new_path.append((first, sec, third, forth))

    elif path_type == "UCC":
        for i in malicious_path:
            first = i[0] + model.start["User"]
            sec = i[1] + model.start["Computer"]
            third = i[2] + model.start["Computer"]
            new_path.append((first, sec, third))
    return new_path



# if auth = 


path, start_idx, end_idx = generate_metapath_sampling(100000) 

# construct C->U->C path






def train(epoch, log_steps=100, eval_steps=2000):
    model.train()
    # print(model._pos_sample())
    total_loss = 0
    final_loss = 0
    for i, (pos_rw, neg_rw) in enumerate(loader):
        optimizer.zero_grad()

        loss = model.loss(pos_rw.to(device), neg_rw.to(device))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        if (i + 1) % log_steps == 0:
            print((f'Epoch: {epoch}, Step: {i + 1:05d}/{len(loader)}, '
                        f'Loss: {total_loss / log_steps:.4f}'))
            final_loss = total_loss/log_steps

            total_loss = 0
    print(final_loss)
    return final_loss
        # if (i + 1) % eval_steps == 0:
        #     acc = test()
        #     print((f'Epoch: {epoch}, Step: {i + 1:05d}/{len(loader)}, '
        #            f'Acc: {acc:.4f}'))


if debug_mode == True:
    train(1)
else:
    start_time = time.time()
    lowest_loss = 10000
    best_model = None
    for epoch in range(1, epoch):
        loss = train(epoch)
        if loss < lowest_loss:
            best_model = model
            lowest_loss = loss
        print("--- %s seconds ---" % (time.time() - start_time))
    # acc = test()
    # print(f'Epoch: {epoch}, Accuracy: {acc:.4f}')

# method 1: get embedding from the skip-gram
# out = []
# for i in range(len(path)):
#     out.append(model.get_embedding(path[i]))
# print(out[1].size())



# mix_graph = True # mix graph = True, the malicious path will be sample from the whole graph 
# mix_graph = False, path will only be sampled from the red team file.

# if mix_graph == False:

dir_normal_path = "/normal_path_" + metapath_strat + ".pt"
normal_path = list(torch.load(dir_save + dir_normal_path))
normal_path = reindexing_path(model, normal_path, metapath_strat)
normal_path_tensor = [torch.LongTensor(i).to(device) for i in normal_path]

print(normal_path_tensor[1])
    
dir_all_malicious_path = "/all_malicious_path_" + metapath_strat + ".pt"
all_malicious_path = list(torch.load(dir_save + dir_all_malicious_path))
all_malicious_path = reindexing_path(model, all_malicious_path, metapath_strat)
all_malicious_path_tensor = [torch.LongTensor(i).to(device) for i in all_malicious_path]

# print(all_malicious_path_tensor)

dir_train_val_malicious_path = "/train_val_malicious_path_" + metapath_strat + ".pt"
train_val_malicious_path = list(torch.load(dir_save + dir_train_val_malicious_path))
train_val_malicious_path = reindexing_path(model, train_val_malicious_path, metapath_strat)
train_val_malicious_path_tensor = [torch.LongTensor(i).to(device) for i in train_val_malicious_path]


dir_test_malicious_path = "/test_malicious_path_" + metapath_strat + ".pt"
test_malicious_path = list(torch.load(dir_save + dir_test_malicious_path))
test_malicious_path = reindexing_path(model, test_malicious_path, metapath_strat)
test_malicious_path_tensor = [torch.LongTensor(i).to(device) for i in test_malicious_path]

# print(malicious_path)
labels = [0 for i in range(len(path))] + [1 for i in range(len(all_malicious_path))]
path = path + all_malicious_path_tensor
print(len(all_malicious_path_tensor))
print(len(path))
# else:





out = dict()
for i in range(len(path)):
    out[path[i]] = model.get_embedding(path[i])
print(out[path[1]].size())
# print(out[1])



out_normal = dict()
for i in range(len(normal_path)):
    out_normal[normal_path[i]] = model.get_embedding(normal_path_tensor[i])


out_mal_train_val = dict()
for i in range(len(train_val_malicious_path)):
    out_mal_train_val[train_val_malicious_path[i]] = model.get_embedding(train_val_malicious_path_tensor[i])


out_mal_test = dict()
for i in range(len(test_malicious_path)):
    out_mal_test[test_malicious_path[i]] = model.get_embedding(test_malicious_path_tensor[i])











# # method 2
# out2 = []
#     # step 1: take node embedding of nodes from a given path 
# computer_emb = model.forward("Computer")
# print(computer_emb.size())
# user_emb = model.forward("User")
# print(user_emb.size())



# # step 2: take mean/max/concatenate, similarity between 2 node to take the edge feature



# for i in range(len(path)):
#     # print(i)
#     # print(path[i])
#     # print(model.start["User"])
#     # print(path[i][1])
#     path_emb = torch.stack((computer_emb[path[i][0] - model.start["Computer"]], user_emb[path[i][1] - model.start["User"]], 
#                                                         computer_emb[path[i][2] - model.start["Computer"]]),-1)
#     # out.append(torch.mean())
#     out2.append(torch.mean(path_emb, 1))











datestring = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
cur_dir = os.getcwd()
store_directory = cur_dir + "/model_" + metapath_strat + "_"+ datestring
os.mkdir(store_directory)





torch.save(best_model, store_directory + '/model.pt')
torch.save(path, store_directory + '/path.pt')
torch.save(out, store_directory + '/path_embedding.pt')
# torch.save(out2, store_directory + '/path_embedding_2.pt')
torch.save(labels, store_directory + '/path_labels.pt')

torch.save(out_normal, store_directory + '/out_normal.pt')
torch.save(out_mal_train_val, store_directory + '/out_mal_train_val.pt')
torch.save(out_mal_test, store_directory + '/out_mal_test.pt')

# method 2: similarity based to construct edge
# print(model.start)
# print(model.end)
# print(torch.cat(path, dim=0))
# print(torch.transpose(torch.cat(path, dim=0), 0, 1))




# for epoch in range(1, 6):
#     train(epoch)
    # acc = test()
    # print(f'Epoch: {epoch}, Accuracy: {acc:.4f}')


# @torch.no_grad()
# def test(train_ratio=0.1):
#     model.eval()

#     z = model('paper', batch=data['paper'].y_index.to(device))
#     print(z.size())
#     print(z)
#     y = data['author'].y

#     perm = torch.randperm(z.size(0))
#     train_perm = perm[:int(z.size(0) * train_ratio)]
#     test_perm = perm[int(z.size(0) * train_ratio):]

#     return model.test(z[train_perm], y[train_perm], z[test_perm], y[test_perm],
#                       max_iter=150)

# generate embedding

