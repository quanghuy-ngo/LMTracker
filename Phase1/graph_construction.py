import pandas as pd
# import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import dgl
import torch
import torch.nn as nn
from csv import reader
from torch_geometric.data import HeteroData
import time
import os, datetime

computer2nodeid = dict()
user2nodeid = dict()
process2nodeid = dict()
ceiling_day = 30

def auth_filter(row, day_filter):
# day_filter: list of day that used to construct graph. Return True when current row in this list

    day_in_second = 60*60*24
    if np.ceil(int(row[0]) / day_in_second) not in day_filter or row[8] == "Fail" or row[7] == "LogOff" or row[1][0] == "C" or row[2][0] == "C":
        return True
    return False
def process_filter(row, day_filter):
# day_filter: list of day that used to construct graph. Return True when current row in this list

    day_in_second = 60*60*24
    if np.ceil(int(row[0]) / day_in_second) not in day_filter or row[4] == "End":
        return True
    return False



print("---------Import auth.txt File --------------")
start_time = time.time()
# open file in read mode
count = 0
day_in_second = 60*60*24
day_filter = [2, 3, 6, 7, 8, 9, 10, 13, 14, 15, 16, 21, 22, 23, 27, 28, 29, 30] # only get data from this list of day
train_val_day = [2, 3, 6, 7, 8, 9, 10]
test_day = [13, 14, 15, 16, 21, 22, 23, 27, 28, 29, 30]
user_comp_dict = set()
comp_comp_dict = set()
user_user_dict = set()
normal_path = set()
with open('/home/andrewngo/Desktop/LANL_data/auth.txt', 'r') as read_obj:
    # pass the file object to reader() to get the reader object
    csv_reader = reader(read_obj)
    # Iterate over each row in the csv using reader object
    for row in csv_reader:
        if auth_filter(row, day_filter):
            continue
        if np.ceil(int(row[0]) / day_in_second) > ceiling_day:
            break
        count = count + 1
        if count % 10000000 == 0:
            print("--- %s seconds ---" % (time.time() - start_time))
            print(str(count/1000000) + "Mth lines")
        source_user     = row[1].split("@")[0]
        des_user        = row[2].split("@")[0]
        source_computer = row[3]
        des_computer    = row[4]
        if source_user not in user2nodeid:
            user2nodeid[source_user] = len(user2nodeid) 
        if des_user not in user2nodeid:
            user2nodeid[des_user] = len(user2nodeid) 
        if source_computer not in computer2nodeid:
            computer2nodeid[source_computer] = len(computer2nodeid) 
        if des_computer not in computer2nodeid:
            computer2nodeid[des_computer] = len(computer2nodeid)
# Comp -> comp edge
        if source_computer != des_computer:
            comp_comp_dict.add((computer2nodeid[source_computer], computer2nodeid[des_computer]))
# user -> comp edge
        user_comp_dict.add((user2nodeid[source_user], computer2nodeid[source_computer]))
        user_comp_dict.add((user2nodeid[des_user], computer2nodeid[des_computer]))
# user -> user edge
        if source_user != des_user:
            user_user_dict.add((user2nodeid[source_user], user2nodeid[des_user]))
        else:
            # sample CUC benign path from log file
            normal_path.add((computer2nodeid[source_computer], user2nodeid[source_user], computer2nodeid[des_computer]))    
# df = pd.DataFrame (df)
# df


from csv import reader
# open file in read mode
print("---------Import Process File --------------")
start_time = time.time()
count = 0
day_in_second = 60*60*24
day_filter = [2, 3, 6, 7, 8, 9, 10, 13, 14, 15, 16, 21, 22, 23, 27, 28, 29, 30] # only get data from this list of day
df_proc = []
comp_proc_dict = set()
with open('/home/andrewngo/Desktop/LANL_data/proc.txt', 'r') as read_obj:
    # pass the file object to reader() to get the reader object
    csv_reader = reader(read_obj)
    # Iterate over each row in the csv using reader object
    for row in csv_reader:
        if np.ceil(int(row[0]) / day_in_second) > ceiling_day:
            break
        if process_filter(row, day_filter):
            continue
        count = count + 1
        if count % 10000000 == 0:
            print("--- %s seconds ---" % (time.time() - start_time))
            print(str(count/1000000) + "Mth lines")
        # row variable is a list that represents a row in csv
        
        if row[2] not in computer2nodeid:
            computer2nodeid[row[2]] = len(computer2nodeid)
            
        if row[3] not in process2nodeid:
            process2nodeid[row[3]] = len(process2nodeid)   
        
        comp_proc_dict.add((computer2nodeid[row[2]], process2nodeid[row[3]]))

# df_proc = pd.DataFrame (df_proc)
# df_proc
# len(comp_proc_dict) 


print("---------Import Red Team File --------------")
malicious_edge_dict = set()
malicious_train_val_day_path = set()
malicious_test_day_path = set()
malicious_edge_path = set()
unseen_nodes = []
with open('/home/andrewngo/Desktop/LANL_data/redteam.txt', 'r') as read_obj:
    # pass the file object to reader() to get the reader object
    csv_reader = reader(read_obj)
    # Iterate over each row in the csv using reader object
    for row in csv_reader:
        day         = row[0]
        user        = row[1].split("@")[0]
        source_computer = row[2]
        des_computer    = row[3]
        if source_computer not in computer2nodeid:
            unseen_nodes.append(source_computer)
            computer2nodeid[source_computer] = len(computer2nodeid)
            
        if des_computer not in computer2nodeid:
            unseen_nodes.append(des_computer)
            computer2nodeid[des_computer] = len(computer2nodeid)
            
        if user not in computer2nodeid:
            unseen_nodes.append(user)
            user2nodeid[user] = len(user2nodeid)
            
        user_comp_dict.add((user2nodeid[user], computer2nodeid[source_computer]))
        user_comp_dict.add((user2nodeid[user], computer2nodeid[des_computer]))
        
        malicious_edge_dict.add((user2nodeid[user], computer2nodeid[source_computer]))
        malicious_edge_dict.add((user2nodeid[user], computer2nodeid[des_computer]))
        malicious_edge_path.add((computer2nodeid[source_computer], user2nodeid[user], computer2nodeid[des_computer]))
        if np.ceil(int(row[0]) / day_in_second) in train_val_day:
            malicious_train_val_day_path.add((computer2nodeid[source_computer], user2nodeid[user], computer2nodeid[des_computer]))
        if np.ceil(int(row[0]) / day_in_second) in test_day:
            malicious_test_day_path.add((computer2nodeid[source_computer], user2nodeid[user], computer2nodeid[des_computer]))



print("------------Constructing Graph---------------")
start_time = time.time()
data_dict = dict()
data_dict[('Computer', 'Connect', 'Computer')] = list(comp_comp_dict)
data_dict[('User', 'Logon', 'Computer')] = list(user_comp_dict)
data_dict[('User', 'SwitchUser', 'User')] = list(user_user_dict)
data_dict[('Computer', 'Create', 'Process')] = list(comp_proc_dict)


g = dgl.heterograph(data_dict)
print(g)

# print(comp_comp_dict)
# gasdhjkasdgkhjas

data_dict[('Computer', 'Connect', 'Computer')] = torch.transpose(torch.LongTensor(data_dict[('Computer', 'Connect', 'Computer')]), 0, 1) 

data_dict[('User', 'Logon', 'Computer')] = torch.transpose(torch.LongTensor(data_dict[('User', 'Logon', 'Computer')]), 0, 1) 

data_dict[('User', 'SwitchUser', 'User')] = torch.transpose(torch.LongTensor(data_dict[('User', 'SwitchUser', 'User')]), 0, 1) 

data_dict[('Computer', 'Create', 'Process')] = torch.transpose(torch.LongTensor(data_dict[('Computer', 'Create', 'Process')]), 0, 1) 



data = HeteroData()
data['Computer', 'Connect', 'Computer'].edge_index = data_dict[('Computer', 'Connect', 'Computer')]
data['User', 'Logon', 'Computer'].edge_index = data_dict[('User', 'Logon', 'Computer')]
data['Computer', 'Logon_rev', 'User'].edge_index = torch.flip(data_dict[('User', 'Logon', 'Computer')], [0, 1]) # flip edge
data['User', 'SwitchUser', 'User'].edge_index = data_dict[('User', 'SwitchUser', 'User')]
data['Computer', 'Create', 'Process'].edge_index = data_dict[('Computer', 'Create', 'Process')]
data['Process', 'Create_rev', 'Computer'].edge_index = torch.flip(data_dict[('Computer', 'Create', 'Process')], [0, 1])# flip edge
print("--- %s seconds ---" % (time.time() - start_time))


print("------------Storing Graph---------------")
start_time = time.time()

datestring = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
cur_dir = os.getcwd()
store_directory = cur_dir + "/graph_data_" + datestring
os.mkdir(store_directory)

dgl.save_graphs(store_directory + "/graph_data_dgl.bin",[g])
torch.save(data, store_directory + '/graph_data_torch.pt')
torch.save(normal_path, store_directory + '/normal_path.pt')
torch.save(malicious_edge_path, store_directory + '/malicious_CUC_path.pt')
torch.save(malicious_train_val_day_path, store_directory + '/malicious_train_val_path.pt')
torch.save(malicious_test_day_path, store_directory + '/malicious_test_path.pt')
torch.save(computer2nodeid, store_directory + '/computer2nodeid.pt')
torch.save(user2nodeid, store_directory + '/user2nodeid.pt')
torch.save(process2nodeid, store_directory + '/process2nodeid.pt')
print("--- %s seconds ---" % (time.time() - start_time))
