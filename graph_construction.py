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
import pickle

computer2nodeid = dict()
user2nodeid = dict()
process2nodeid = dict()
auth_type2nodeid = dict()
logon_type2nodeid = dict()
logon_orient2nodeid = dict()
ceiling_day = 30

def auth_filter(row, day_filter):
# day_filter: list of day that used to construct graph. Return True when current row in this list

    day_in_second = 60*60*24
    # if np.ceil(int(row[0]) / day_in_second) not in day_filter or row[8] == "Fail" or row[7] == "LogOff" or row[1][0] == "C" or row[2][0] == "C":
    if np.ceil(int(row[0]) / day_in_second) not in day_filter or row[8] == "Fail" or row[1][0] == "C" or row[2][0] == "C":
        return True
    return False
def process_filter(row, day_filter):
# day_filter: list of day that used to construct graph. Return True when current row in this list

    day_in_second = 60*60*24
    if np.ceil(int(row[0]) / day_in_second) not in day_filter or row[4] == "End":
        return True
    return False



# open red_red_team.pkl 
with open("/home/andrewngo/Desktop/LANL_data/red_team_list.pkl", "rb") as fp:   # Unpickling
    red_team_line = pickle.load(fp)
red_team_line = set(red_team_line)




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
comp_auth_type_dict  = set()
comp_logon_type_dict  = set()
comp_logon_orient_dict  = set()

normal_path_CUC = set()
normal_path_UCC = set()
normal_path_UCAC = set()
normal_path_UCCA = set()

with open('/home/andrewngo/Desktop/LANL_data/auth.txt', 'r') as read_obj:
    # pass the file object to reader() to get the reader object
    csv_reader = reader(read_obj)
    # Iterate over each row in the csv using reader object
    for row in csv_reader:

        count = count + 1
        # skip redteam sample
        if count in red_team_line:
            continue



        if count % 10000000 == 0:
            print("--- %s seconds ---" % (time.time() - start_time))
            print(str(count/1000000) + "Mth lines")
        
        
        
        if auth_filter(row, day_filter):
            continue
        if np.ceil(int(row[0]) / day_in_second) > ceiling_day:
            break


        source_user     = row[1].split("@")[0]
        des_user        = row[2].split("@")[0]
        source_computer = row[3]
        des_computer    = row[4]
        auth_type       = row[5]
        logon_type      = row[6] # 
        logon_orient    = row[7] # logon, logoff
        
        if "MICROSOFT" in auth_type:
            auth_type = "MICROSOFT_AUTHENTICATION_PACKAGE_V1_0"
            
            
# update node dictionary   
        
        if source_user not in user2nodeid:
            user2nodeid[source_user] = len(user2nodeid) 
        if des_user not in user2nodeid:
            user2nodeid[des_user] = len(user2nodeid) 
        if source_computer not in computer2nodeid:
            computer2nodeid[source_computer] = len(computer2nodeid) 
        if des_computer not in computer2nodeid:
            computer2nodeid[des_computer] = len(computer2nodeid)
        if auth_type not in auth_type2nodeid:
            auth_type2nodeid[auth_type] = len(auth_type2nodeid)
        if logon_type not in logon_type2nodeid:
            logon_type2nodeid[logon_type] = len(logon_type2nodeid)
        if logon_orient not in logon_orient2nodeid:
            logon_orient2nodeid[logon_orient] = len(logon_orient2nodeid)    
            
# ***********Add more edge type here**********            
# note that we consider undirecterd graph for metapath2vec

# Comp -> comp edge
        if source_computer != des_computer:
            comp_comp_dict.add((computer2nodeid[source_computer], computer2nodeid[des_computer]))
            comp_comp_dict.add((computer2nodeid[des_computer], computer2nodeid[source_computer]))
# user -> comp edge
        user_comp_dict.add((user2nodeid[source_user], computer2nodeid[source_computer]))
        user_comp_dict.add((user2nodeid[des_user], computer2nodeid[des_computer]))
# user -> user edge
        if source_user != des_user:
            user_user_dict.add((user2nodeid[source_user], user2nodeid[des_user]))
        else:
            # sample CUC benign path from log file
            normal_path_CUC.add((computer2nodeid[source_computer], user2nodeid[source_user], computer2nodeid[des_computer])) 
            normal_path_UCC.add((user2nodeid[source_user], computer2nodeid[source_computer], computer2nodeid[des_computer])) 
            normal_path_UCAC.add((user2nodeid[source_user], computer2nodeid[source_computer], auth_type2nodeid[auth_type], computer2nodeid[des_computer]))    
            normal_path_UCCA.add((user2nodeid[source_user], computer2nodeid[source_computer], computer2nodeid[des_computer], auth_type2nodeid[auth_type]))
# auth_type -> comp 
        comp_auth_type_dict.add((computer2nodeid[source_computer], auth_type2nodeid[auth_type]))
        comp_auth_type_dict.add((computer2nodeid[des_computer], auth_type2nodeid[auth_type]))
        
        comp_logon_type_dict.add((computer2nodeid[source_computer], logon_type2nodeid[logon_type]))
        comp_logon_type_dict.add((computer2nodeid[des_computer], logon_type2nodeid[logon_type]))
        
        comp_logon_orient_dict.add((computer2nodeid[source_computer], logon_orient2nodeid[logon_orient]))
        comp_logon_orient_dict.add((computer2nodeid[des_computer], logon_orient2nodeid[logon_orient]))
            


# from csv import reader
# # open file in read mode
# print("---------Import Process File --------------")
# start_time = time.time()
# count = 0
# day_in_second = 60*60*24
# day_filter = [2, 3, 6, 7, 8, 9, 10, 13, 14, 15, 16, 21, 22, 23, 27, 28, 29, 30] # only get data from this list of day
# df_proc = []
# comp_proc_dict = set()
# with open('/home/andrewngo/Desktop/LANL_data/proc.txt', 'r') as read_obj:
#     # pass the file object to reader() to get the reader object
#     csv_reader = reader(read_obj)
#     # Iterate over each row in the csv using reader object
#     for row in csv_reader:
#         if np.ceil(int(row[0]) / day_in_second) > ceiling_day:
#             break
#         if process_filter(row, day_filter):
#             continue
#         count = count + 1
#         if count % 10000000 == 0:
#             print("--- %s seconds ---" % (time.time() - start_time))
#             print(str(count/1000000) + "Mth lines")
#         # row variable is a list that represents a row in csv
        
#         if row[2] not in computer2nodeid:
#             computer2nodeid[row[2]] = len(computer2nodeid)
            
#         if row[3] not in process2nodeid:
#             process2nodeid[row[3]] = len(process2nodeid)   
        
#         comp_proc_dict.add((computer2nodeid[row[2]], process2nodeid[row[3]]))

# df_proc = pd.DataFrame (df_proc)
# df_proc
# len(comp_proc_dict) 


print("---------Import Red Team File --------------")
malicious_edge_dict = set()

all_malicious_path_CUC = set()
all_malicious_path_UCC = set()
all_malicious_path_UCAC = set()
all_malicious_path_UCCA = set()

train_val_malicious_path_CUC = set()
train_val_malicious_path_UCC = set()
train_val_malicious_path_UCAC = set()
train_val_malicious_path_UCCA = set()

test_malicious_path_CUC = set()
test_malicious_path_UCC = set()
test_malicious_path_UCAC = set()
test_malicious_path_UCCA = set()



unseen_nodes = []
with open('/home/andrewngo/Desktop/LANL_data/redteam_auth.txt', 'r') as read_obj:
    # pass the file object to reader() to get the reader object
    csv_reader = reader(read_obj)
    # Iterate over each row in the csv using reader object
    for row in csv_reader:
        source_user     = row[1].split("@")[0]
        des_user        = row[2].split("@")[0]
        source_computer = row[3]
        des_computer    = row[4]
        auth_type       = row[5]
        logon_type      = row[6] # 
        logon_orient    = row[7] # logon, logoff
        
        
        if "MICROSOFT" in auth_type:
            auth_type = "MICROSOFT_AUTHENTICATION_PACKAGE_V1_0"
            
            
# update node dictionary   
        
        if source_user not in user2nodeid:
            user2nodeid[source_user] = len(user2nodeid) 
        if des_user not in user2nodeid:
            user2nodeid[des_user] = len(user2nodeid) 
        if source_computer not in computer2nodeid:
            computer2nodeid[source_computer] = len(computer2nodeid) 
        if des_computer not in computer2nodeid:
            computer2nodeid[des_computer] = len(computer2nodeid)
        if auth_type not in auth_type2nodeid:
            auth_type2nodeid[auth_type] = len(auth_type2nodeid)
        if logon_type not in logon_type2nodeid:
            logon_type2nodeid[logon_type] = len(logon_type2nodeid)
        if logon_orient not in logon_orient2nodeid:
            logon_orient2nodeid[logon_orient] = len(logon_orient2nodeid)   
            
            
            
        # user_comp_dict.add((user2nodeid[source_user], computer2nodeid[source_computer]))
        # user_comp_dict.add((user2nodeid[des_user], computer2nodeid[des_computer]))
        
        malicious_edge_dict.add((user2nodeid[source_user], computer2nodeid[source_computer]))
        malicious_edge_dict.add((user2nodeid[des_user], computer2nodeid[des_computer]))

        
# user -> src comp -> auth_type -> des comp: UCAC
# user -> src comp -> des comp: UCC
# src comp -> user -> des comp: CUC


        all_malicious_path_CUC.add((computer2nodeid[source_computer], user2nodeid[source_user], computer2nodeid[des_computer]))
        all_malicious_path_UCAC.add((user2nodeid[source_user], computer2nodeid[source_computer], auth_type2nodeid[auth_type], computer2nodeid[des_computer]))
        all_malicious_path_UCC.add((user2nodeid[source_user], computer2nodeid[source_computer], computer2nodeid[des_computer]))
        all_malicious_path_UCCA.add((user2nodeid[source_user], computer2nodeid[source_computer], computer2nodeid[des_computer], auth_type2nodeid[auth_type]))
        
        if np.ceil(int(row[0]) / day_in_second) in train_val_day:
            train_val_malicious_path_CUC.add((computer2nodeid[source_computer], user2nodeid[source_user], computer2nodeid[des_computer]))
            train_val_malicious_path_UCAC.add((user2nodeid[source_user], computer2nodeid[source_computer], auth_type2nodeid[auth_type], computer2nodeid[des_computer]))
            train_val_malicious_path_UCC.add((user2nodeid[source_user], computer2nodeid[source_computer], computer2nodeid[des_computer]))        
            train_val_malicious_path_UCCA.add((user2nodeid[source_user], computer2nodeid[source_computer], computer2nodeid[des_computer], auth_type2nodeid[auth_type]))
                                        
                                        
        if np.ceil(int(row[0]) / day_in_second) in test_day:
            test_malicious_path_CUC.add((computer2nodeid[source_computer], user2nodeid[source_user], computer2nodeid[des_computer]))
            test_malicious_path_UCAC.add((user2nodeid[source_user], computer2nodeid[source_computer], auth_type2nodeid[auth_type], computer2nodeid[des_computer]))
            test_malicious_path_UCC.add((user2nodeid[source_user], computer2nodeid[source_computer], computer2nodeid[des_computer]))        
            test_malicious_path_UCAC.add((user2nodeid[source_user], computer2nodeid[source_computer], computer2nodeid[des_computer], auth_type2nodeid[auth_type]))


print("------------Constructing Graph---------------")
start_time = time.time()
data_dict = dict()
data_dict[('Computer', 'Connect', 'Computer')] = list(comp_comp_dict)
data_dict[('User', 'Logon', 'Computer')] = list(user_comp_dict)
data_dict[('User', 'SwitchUser', 'User')] = list(user_user_dict)
# data_dict[('Computer', 'Create', 'Process')] = list(comp_proc_dict)
data_dict[('Computer', 'Use', 'Auth_Type')] = list(comp_auth_type_dict)
data_dict[('Computer', 'Use_logon', 'Logon_type')] = list(comp_logon_type_dict)
data_dict[('Computer', 'Have', 'Logon_orient')] = list(comp_logon_orient_dict)









g = dgl.heterograph(data_dict)
print(g)

# print(comp_comp_dict)
# gasdhjkasdgkhjas

data_dict[('Computer', 'Connect', 'Computer')] = torch.transpose(torch.LongTensor(data_dict[('Computer', 'Connect', 'Computer')]), 0, 1) 

data_dict[('User', 'Logon', 'Computer')] = torch.transpose(torch.LongTensor(data_dict[('User', 'Logon', 'Computer')]), 0, 1) 

data_dict[('User', 'SwitchUser', 'User')] = torch.transpose(torch.LongTensor(data_dict[('User', 'SwitchUser', 'User')]), 0, 1) 

# data_dict[('Computer', 'Create', 'Process')] = torch.transpose(torch.LongTensor(data_dict[('Computer', 'Create', 'Process')]), 0, 1) 

data_dict[('Computer', 'Use', 'Auth_Type')] = torch.transpose(torch.LongTensor(data_dict[('Computer', 'Use', 'Auth_Type')]), 0, 1) 

data_dict[('Computer', 'Use_logon', 'Logon_type')] = torch.transpose(torch.LongTensor(data_dict[('Computer', 'Use_logon', 'Logon_type')]), 0, 1) 

data_dict[('Computer', 'Have', 'Logon_orient')] = torch.transpose(torch.LongTensor(data_dict[('Computer', 'Have', 'Logon_orient')]), 0, 1) 



data = HeteroData()
data['Computer', 'Connect', 'Computer'].edge_index = data_dict[('Computer', 'Connect', 'Computer')]
data['User', 'Logon', 'Computer'].edge_index = data_dict[('User', 'Logon', 'Computer')]

data['Computer', 'Logon_rev', 'User'].edge_index = torch.flip(data_dict[('User', 'Logon', 'Computer')], [0, 1]) # flip edge
data['User', 'SwitchUser', 'User'].edge_index = data_dict[('User', 'SwitchUser', 'User')]

# data['Computer', 'Create', 'Process'].edge_index = data_dict[('Computer', 'Create', 'Process')]
# data['Process', 'Create_rev', 'Computer'].edge_index = torch.flip(data_dict[('Computer', 'Create', 'Process')], [0, 1])# flip edge

data['Computer', 'Use', 'Auth_Type'].edge_index = data_dict[('Computer', 'Use', 'Auth_Type')]
data['Auth_Type', 'Use_rev','Computer'].edge_index = torch.flip(data_dict[('Computer', 'Use', 'Auth_Type')], [0, 1])# flip edge

data['Computer', 'Use_logon', 'Logon_type'].edge_index = data_dict[('Computer', 'Use_logon', 'Logon_type')]
data['Logon_type', 'Use_logon_rev', 'Computer'].edge_index = torch.flip(data_dict[('Computer', 'Use_logon', 'Logon_type')], [0, 1])# flip edge

data['Computer', 'Have', 'Logon_orient'].edge_index = data_dict[('Computer', 'Have', 'Logon_orient')]
data['Logon_orient', 'Have_rev', 'Computer'].edge_index = torch.flip(data_dict[('Computer', 'Have', 'Logon_orient')], [0, 1])# flip edge





print("--- %s seconds ---" % (time.time() - start_time))


print("------------Storing Graph---------------")
start_time = time.time()

datestring = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
cur_dir = os.getcwd()
store_directory = cur_dir + "/graph_data_" + datestring


print("Graph store at: " + store_directory)

os.mkdir(store_directory)

dgl.save_graphs(store_directory + "/graph_data_dgl.bin",[g])
torch.save(data, store_directory + '/graph_data_torch.pt')
torch.save(normal_path_CUC, store_directory + '/normal_path_CUC.pt')
torch.save(normal_path_UCC, store_directory + '/normal_path_UCC.pt')
torch.save(normal_path_UCAC, store_directory + '/normal_path_UCAC.pt')
torch.save(normal_path_UCCA, store_directory + '/normal_path_UCCA.pt')



torch.save(all_malicious_path_CUC, store_directory + '/all_malicious_path_CUC.pt')
torch.save(all_malicious_path_UCC, store_directory + '/all_malicious_path_UCC.pt')
torch.save(all_malicious_path_UCAC, store_directory + '/all_malicious_path_UCAC.pt')
torch.save(all_malicious_path_UCCA, store_directory + '/all_malicious_path_UCCA.pt')

torch.save(train_val_malicious_path_CUC, store_directory + '/train_val_malicious_path_CUC.pt')
torch.save(train_val_malicious_path_UCC, store_directory + '/train_val_malicious_path_UCC.pt')
torch.save(train_val_malicious_path_UCAC, store_directory + '/train_val_malicious_path_UCAC.pt')
torch.save(train_val_malicious_path_UCCA, store_directory + '/train_val_malicious_path_UCCA.pt')

torch.save(test_malicious_path_CUC, store_directory + '/test_malicious_path_CUC.pt')
torch.save(test_malicious_path_UCC, store_directory + '/test_malicious_path_UCC.pt')
torch.save(test_malicious_path_UCAC, store_directory + '/test_malicious_path_UCAC.pt')
torch.save(test_malicious_path_UCCA, store_directory + '/test_malicious_path_UCCA.pt')

torch.save(computer2nodeid, store_directory + '/computer2nodeid.pt')
torch.save(user2nodeid, store_directory + '/user2nodeid.pt')
torch.save(process2nodeid, store_directory + '/process2nodeid.pt')
print("--- %s seconds ---" % (time.time() - start_time))
