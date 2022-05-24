# Importing Libraries
from tabnanny import verbose
import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
# import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
import torch
from numpy.random import permutation
from sklearn.metrics import roc_curve, roc_auc_score
# from matplotlib import pyplot
from numpy import sqrt, argmax
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, precision_recall_curve

embedding_dir = "/home/andrewngo/Desktop/MLTracker/model_UCAC_20220303110748" # model folder
path = torch.load(embedding_dir + "/path.pt")
path_embedding = torch.load(embedding_dir + "/path_embedding.pt")
labels = torch.load(embedding_dir + "/path_labels.pt")
out_mal_train_val_dict = torch.load(embedding_dir + "/out_mal_train_val.pt")
out_mal_test_dict = torch.load(embedding_dir + "/out_mal_test.pt")
out_normal_dict = torch.load(embedding_dir + "/out_normal.pt")


path_embedding = pd.DataFrame(list(path_embedding.values())).astype(float)
out_mal_test_keys = list(out_mal_test_dict.keys())
out_mal_val_keys = list(out_mal_train_val_dict.keys())
out_normal_keys = list(out_normal_dict.keys())




# Test 1
# Random shuffling testing strategy with benign sample sampling from graph (Mentioned in paper)
def graph_sample_benign_split(path_embedding, out_mal_test_keys, out_mal_val_keys, out_normal_keys, labels, training_sample = 10000, val_normal_sample = 5000, test_normal_sample = 5000, seed = 10):

    benign_path = path_embedding.values[:labels.count(0)]
    malicious_path = path_embedding.values[labels.count(0):]


    training_sample = 10000
    val_normal_sample = 5000
    test_normal_sample = 5000

    #validation + testing = 1 for easier code, the correct ratio value should be validation + testing + training = 1
    validation_ratio = 0.5
    test_ratio = 0.5

    np.random.seed(seed)
    perm = permutation(len(benign_path))
    # normal_train_data_idx = 
    normal_train_data_idx = perm[:training_sample]

    perm1 = permutation(len(out_normal_keys))

    perm_val = perm1[:val_normal_sample]
    perm_test = perm1[val_normal_sample:(val_normal_sample+test_normal_sample)]


    perm = permutation(len(malicious_path))
    mal_val_data_idx = perm[:int(len(perm)*test_ratio)]
    mal_test_data_idx = perm[int(len(perm)*test_ratio):]

    normal_train_data = np.asarray([benign_path[i] for i in normal_train_data_idx])

    out_test_keys_sample = [out_normal_keys[i] for i in perm_test]
    normal_test_data = np.asarray([out_normal_dict[i].tolist() for i in out_test_keys_sample])

    out_val_keys_sample = [out_normal_keys[i] for i in perm_val]
    normal_val_data = np.asarray([out_normal_dict[i].tolist() for i in out_val_keys_sample])


    mal_val_data = np.asarray([malicious_path[i] for i in mal_val_data_idx])
    mal_test_data = np.asarray([malicious_path[i] for i in mal_test_data_idx])


    # print(normal_train_data)
    # Initializing a MinMax Scaler
    scaler = MinMaxScaler()

    # Fitting the train data to the scaler
    data_scaled = scaler.fit(normal_train_data)




    normal_train_data = data_scaled.transform(normal_train_data)
    normal_val_data = data_scaled.transform(normal_val_data)
    normal_test_data = data_scaled.transform(normal_test_data)
    mal_val_data = data_scaled.transform(mal_val_data)
    mal_test_data = data_scaled.transform(mal_test_data)

    # normal_train_data

    test_data = np.concatenate((normal_test_data,mal_test_data), axis=0)
    val_data =  np.concatenate((normal_val_data, mal_val_data), axis=0)
    labels_test = [0 for i in range(len(normal_test_data))] + [1 for i in range(len(mal_test_data))]
    labels_val = [0 for i in range(len(normal_val_data))] + [1 for i in range(len(mal_val_data))]

    return normal_train_data, normal_val_data, normal_test_data, mal_val_data, mal_test_data, test_data, val_data, labels_test, labels_val




# Test 2
def log_sample_bengin_split(path_embedding, out_mal_test_keys, out_mal_val_keys, out_normal_keys, labels, training_sample = 10000, val_normal_sample = 5000, test_normal_sample = 5000, seed = 10):

    benign_path = path_embedding.values[:labels.count(0)]
    malicious_path = path_embedding.values[labels.count(0):]


    #validation + testing = 1 for easier code, the correct ratio value should be validation + testing + training = 1
    validation_ratio = 0.5
    test_ratio = 0.5
    np.random.seed(seed)
    perm = permutation(len(out_normal_keys))
    # normal_train_data_idx = 

    perm_val = perm[:val_normal_sample]
    perm_test = perm[val_normal_sample:(val_normal_sample+test_normal_sample)]
    perm_train = perm[(val_normal_sample+test_normal_sample):(val_normal_sample+test_normal_sample+training_sample)]

    perm = permutation(len(malicious_path))
    mal_val_data_idx = perm[:int(len(perm)*test_ratio)]
    mal_test_data_idx = perm[int(len(perm)*test_ratio):]


    # normal_train_data_idx
    # normal_val_data_idx
    # normal_test_data_idx
    # mal_val_data_idx
    # mal_test_data_idx

    out_train_keys_sample = [out_normal_keys[i] for i in perm_train]
    normal_train_data = np.asarray([out_normal_dict[i].tolist() for i in out_train_keys_sample])

    out_test_keys_sample = [out_normal_keys[i] for i in perm_test]
    normal_test_data = np.asarray([out_normal_dict[i].tolist() for i in out_test_keys_sample])

    out_val_keys_sample = [out_normal_keys[i] for i in perm_val]
    normal_val_data = np.asarray([out_normal_dict[i].tolist() for i in out_val_keys_sample])

    mal_val_data = np.asarray([malicious_path[i] for i in mal_val_data_idx])
    mal_test_data = np.asarray([malicious_path[i] for i in mal_test_data_idx])


    # print(normal_train_data)
    # Initializing a MinMax Scaler
    scaler = MinMaxScaler()

    # Fitting the train data to the scaler
    data_scaled = scaler.fit(normal_train_data)




    normal_train_data = data_scaled.transform(normal_train_data)
    normal_val_data = data_scaled.transform(normal_val_data)
    normal_test_data = data_scaled.transform(normal_test_data)
    mal_val_data = data_scaled.transform(mal_val_data)
    mal_test_data = data_scaled.transform(mal_test_data)

    # normal_train_data

    test_data = np.concatenate((normal_test_data,mal_test_data), axis=0)
    val_data =  np.concatenate((normal_val_data, mal_val_data), axis=0)
    labels_test = [0 for i in range(len(normal_test_data))] + [1 for i in range(len(mal_test_data))]
    labels_val = [0 for i in range(len(normal_val_data))] + [1 for i in range(len(mal_val_data))]
# normal_train_data, normal_test_data, train_labels, test_labels = train_test_split(benign_path, , test_size = 0.2, random_state = 111)
    return normal_train_data, normal_val_data, normal_test_data, mal_val_data, mal_test_data, test_data, val_data, labels_test, labels_val




# Test 3
def graph_sample_bengin_day_split(path_embedding, out_mal_test_keys, out_mal_val_keys, out_normal_keys, labels, training_sample = 10000, val_normal_sample = 5000, test_normal_sample = 5000, seed = 10):
    
    benign_path = path_embedding.values[:labels.count(0)]
    malicious_path = path_embedding.values[labels.count(0):]
    np.random.seed(seed)
    perm = permutation(len(out_normal_keys))

    perm_val = perm[:val_normal_sample]
    perm_test = perm[(val_normal_sample):(val_normal_sample+test_normal_sample)]
    perm_train = perm[(val_normal_sample+test_normal_sample):(training_sample+val_normal_sample+test_normal_sample)]

    perm = permutation(len(benign_path))
    # normal_train_data_idx = 
    normal_train_data_idx = perm[:training_sample]

    # out_train_keys_sample = [out_normal_keys[i] for i in perm_train]
    # normal_train_data = np.asarray([out_normal_dict[i].tolist() for i in out_train_keys_sample])
    normal_train_data = np.asarray([benign_path[i] for i in normal_train_data_idx])


    out_test_keys_sample = [out_normal_keys[i] for i in perm_test]
    normal_test_data = np.asarray([out_normal_dict[i].tolist() for i in out_test_keys_sample])

    out_val_keys_sample = [out_normal_keys[i] for i in perm_val]
    normal_val_data = np.asarray([out_normal_dict[i].tolist() for i in out_val_keys_sample])


    mal_val_data = np.asarray([out_mal_train_val_dict[i].tolist() for i in out_mal_train_val_dict])
    mal_test_data = np.asarray([out_mal_test_dict[i].tolist() for i in out_mal_test_dict])

    scaler = MinMaxScaler()

    # Fitting the train data to the scaler
    data_scaled = scaler.fit(normal_train_data)

    normal_train_data = data_scaled.transform(normal_train_data)
    normal_val_data = data_scaled.transform(normal_val_data)
    normal_test_data = data_scaled.transform(normal_test_data)
    mal_val_data = data_scaled.transform(mal_val_data)
    mal_test_data = data_scaled.transform(mal_test_data)

    test_data = np.concatenate((normal_test_data,mal_test_data), axis=0)
    val_data =  np.concatenate((normal_val_data, mal_val_data), axis=0)
    labels_test = [0 for i in range(len(normal_test_data))] + [1 for i in range(len(mal_test_data))]
    labels_val = [0 for i in range(len(normal_val_data))] + [1 for i in range(len(mal_val_data))]

# normal_train_data, normal_test_data, train_labels, test_labels = train_test_split(benign_path, , test_size = 0.2, random_state = 111)
    return normal_train_data, normal_val_data, normal_test_data, mal_val_data, mal_test_data, test_data, val_data, labels_test, labels_val



# Test 4
def log_sample_bengin_day_split(path_embedding, out_mal_test_keys, out_mal_val_keys, out_normal_keys, labels, training_sample = 10000, val_normal_sample = 5000, test_normal_sample = 5000, seed = 10):

    np.random.seed(seed)
    perm = permutation(len(out_normal_keys))

    perm_val = perm[:val_normal_sample]
    perm_test = perm[(val_normal_sample):(val_normal_sample+test_normal_sample)]
    perm_train = perm[(val_normal_sample+test_normal_sample):(training_sample+val_normal_sample+test_normal_sample)]


    out_train_keys_sample = [out_normal_keys[i] for i in perm_train]
    normal_train_data = np.asarray([out_normal_dict[i].tolist() for i in out_train_keys_sample])

    out_test_keys_sample = [out_normal_keys[i] for i in perm_test]
    normal_test_data = np.asarray([out_normal_dict[i].tolist() for i in out_test_keys_sample])

    out_val_keys_sample = [out_normal_keys[i] for i in perm_val]
    normal_val_data = np.asarray([out_normal_dict[i].tolist() for i in out_val_keys_sample])


    mal_val_data = np.asarray([out_mal_train_val_dict[i].tolist() for i in out_mal_train_val_dict])
    mal_test_data = np.asarray([out_mal_test_dict[i].tolist() for i in out_mal_test_dict])

    scaler = MinMaxScaler()

    # Fitting the train data to the scaler
    data_scaled = scaler.fit(normal_train_data)

    normal_train_data = data_scaled.transform(normal_train_data)
    normal_val_data = data_scaled.transform(normal_val_data)
    normal_test_data = data_scaled.transform(normal_test_data)
    mal_val_data = data_scaled.transform(mal_val_data)
    mal_test_data = data_scaled.transform(mal_test_data)

    test_data = np.concatenate((normal_test_data,mal_test_data), axis=0)
    val_data =  np.concatenate((normal_val_data, mal_val_data), axis=0)
    labels_test = [0 for i in range(len(normal_test_data))] + [1 for i in range(len(mal_test_data))]
    labels_val = [0 for i in range(len(normal_val_data))] + [1 for i in range(len(mal_val_data))]

# normal_train_data, normal_test_data, train_labels, test_labels = train_test_split(benign_path, , test_size = 0.2, random_state = 111)
    return normal_train_data, normal_val_data, normal_test_data, mal_val_data, mal_test_data, test_data, val_data, labels_test, labels_val



class Autoencoder(Model):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = Sequential([
                                Dense(128, activation='tanh'),
#                                 Dense(32, activation='relu'),
                                Dense(64, activation='tanh'),
#                                 Dense(8, activation='relu')
])
        self.decoder = Sequential([
                               Dense(64, activation='tanh'),
#                                Dense(32, activation='rel'),
                               Dense(128, activation='tanh')])

    def call(self,x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


# Instantiating the Autoencoder
model = Autoencoder()

# creating an early_stopping
# early_stopping = EarlyStopping(monitor='val_loss',
#                                patience = 20,
#                                mode = 'min')
early_stopping = EarlyStopping(
    monitor='val_loss',
    min_delta=0.0001,
    patience=20,
    verbose=1, 
    mode='min',
    restore_best_weights=True)
# Compiling the model
model.compile(optimizer = 'adam',
              loss = 'mae')

training_sample = 10000
val_normal_sample = 5000
test_normal_sample = 5000

best_strat = "AUC"
# best_strat = "F1"



# import pandas as pd

# def classification_report_csv(report):
#     report_data = []
#     lines = report.split('\n')
#     for line in lines[2:-3]:
#         row = {}
#         row_data = line.split('      ')
#         row['class'] = row_data[0]
#         row['precision'] = float(row_data[1])
#         row['recall'] = float(row_data[2])
#         row['f1_score'] = float(row_data[3])
#         row['support'] = float(row_data[4])
#         report_data.append(row)
#     dataframe = pd.DataFrame.from_dict(report_data)
#     dataframe.to_csv('classification_report.csv', index = False)




print("______________________TEST 1_________________________")
for i in range(10):
    normal_train_data, normal_val_data, normal_test_data, mal_val_data, mal_test_data, test_data, val_data, labels_test, labels_val = graph_sample_benign_split(path_embedding, out_mal_test_keys, out_mal_val_keys, 
                                out_normal_keys, labels, training_sample = 10000, val_normal_sample = 5000, 
                                test_normal_sample = 5000, seed = i)
    # Training the model
    validate = np.concatenate((normal_train_data, normal_val_data), axis=0)
    history = model.fit(normal_train_data,normal_train_data,
                        epochs = 500,
                        batch_size = 128,
                        validation_data = (validate,validate),
                        shuffle = True,
                        callbacks = [early_stopping], verbose = 0)
    reconstructions_a = model.predict(test_data)
    test_loss = tf.keras.losses.mae(reconstructions_a, test_data)

    prob_test = (test_loss-min(test_loss))/(max(test_loss)-min(test_loss))
    #
    if best_strat == "AUC":
        fpr, tpr, thresholds = roc_curve(labels_test, prob_test)
        gmeans = sqrt(tpr * (1-fpr))
        ix = argmax(gmeans)
        threshold = thresholds[ix]
    elif best_strat == "F1":
        precision, recall, thresholds = precision_recall_curve(labels_test, prob_test)
        f1_scores = 2*recall*precision/(recall+precision)
        f1_scores = np.nan_to_num(f1_scores)
        threshold = thresholds[np.argmax(f1_scores)]
        # print('Best threshold: ', thresholds[np.argmax(f1_scores)])
        # print('Best F1-Score: ', np.max(f1_scores))

    pred_test = [0 if prob_test[i] < threshold else 1 for i in range(len(prob_test))]
    print("Test Evaluation with Seed: " + str(i))
    print("AUC Score: " + str(roc_auc_score(labels_test, pred_test, average=None)))
    print(classification_report(labels_test ,pred_test,  labels=[0, 1], target_names=['benign', 'malicious']))
    print(confusion_matrix(labels_test ,pred_test))




print("______________________TEST 2_________________________")
for i in range(10):
    normal_train_data, normal_val_data, normal_test_data, mal_val_data, mal_test_data, test_data, val_data, labels_test, labels_val = log_sample_bengin_split(path_embedding, out_mal_test_keys, out_mal_val_keys, 
                                out_normal_keys, labels, training_sample = 10000, val_normal_sample = 5000, 
                                test_normal_sample = 5000, seed = i)
    # Training the model
    validate = np.concatenate((normal_train_data, normal_val_data), axis=0)
    history = model.fit(normal_train_data,normal_train_data,
                        epochs = 500,
                        batch_size = 128,
                        validation_data = (validate,validate),
                        shuffle = True,
                        callbacks = [early_stopping], verbose = 0)
    reconstructions_a = model.predict(test_data)
    test_loss = tf.keras.losses.mae(reconstructions_a, test_data)
    
    prob_test = (test_loss-min(test_loss))/(max(test_loss)-min(test_loss))
    if best_strat == "AUC":
        fpr, tpr, thresholds = roc_curve(labels_test, prob_test)
        gmeans = sqrt(tpr * (1-fpr))
        ix = argmax(gmeans)
        threshold = thresholds[ix]
    elif best_strat == "F1":
        precision, recall, thresholds = precision_recall_curve(labels_test, prob_test)
        f1_scores = 2*recall*precision/(recall+precision)
        f1_scores = np.nan_to_num(f1_scores)
        threshold = thresholds[np.argmax(f1_scores)]
    # print(roc_auc_score(labels_test, prob_test, average=None))
    pred_test = [0 if prob_test[i] < threshold else 1 for i in range(len(prob_test))]
    print("Test Evaluation with Seed: " + str(i))
    print("AUC Score: " + str(roc_auc_score(labels_test, pred_test, average=None)))
    print(classification_report(labels_test ,pred_test,  labels=[0, 1], target_names=['benign', 'malicious']))
    print(confusion_matrix(labels_test ,pred_test))




print("______________________TEST 3_________________________")
for i in range(10):
    normal_train_data, normal_val_data, normal_test_data, mal_val_data, mal_test_data, test_data, val_data, labels_test, labels_val = graph_sample_bengin_day_split(path_embedding, out_mal_test_keys, out_mal_val_keys, 
                                out_normal_keys, labels, training_sample = 10000, val_normal_sample = 5000, 
                                test_normal_sample = 5000, seed = i)
    # Training the model
    validate = np.concatenate((normal_train_data, normal_val_data), axis=0)
    history = model.fit(normal_train_data,normal_train_data,
                        epochs = 500,
                        batch_size = 128,
                        validation_data = (validate,validate),
                        shuffle = True,
                        callbacks = [early_stopping], verbose = 0)
    reconstructions_a = model.predict(test_data)
    test_loss = tf.keras.losses.mae(reconstructions_a, test_data)
    
    prob_test = (test_loss-min(test_loss))/(max(test_loss)-min(test_loss))
    if best_strat == "AUC":
        fpr, tpr, thresholds = roc_curve(labels_test, prob_test)
        gmeans = sqrt(tpr * (1-fpr))
        ix = argmax(gmeans)
        threshold = thresholds[ix]
    elif best_strat == "F1":
        precision, recall, thresholds = precision_recall_curve(labels_test, prob_test)
        f1_scores = 2*recall*precision/(recall+precision)
        f1_scores = np.nan_to_num(f1_scores)
        threshold = thresholds[np.argmax(f1_scores)]
    # print(roc_auc_score(labels_test, prob_test, average=None))
    pred_test = [0 if prob_test[i] < threshold else 1 for i in range(len(prob_test))]
    print("Test Evaluation with Seed: " + str(i))
    print("AUC Score: " + str(roc_auc_score(labels_test, pred_test, average=None)))
    print(classification_report(labels_test ,pred_test,  labels=[0, 1], target_names=['benign', 'malicious']))
    print(confusion_matrix(labels_test ,pred_test))






print("______________________TEST 4_________________________")
for i in range(10):
    normal_train_data, normal_val_data, normal_test_data, mal_val_data, mal_test_data, test_data, val_data, labels_test, labels_val = log_sample_bengin_day_split(path_embedding, out_mal_test_keys, out_mal_val_keys, 
                                out_normal_keys, labels, training_sample = 10000, val_normal_sample = 5000, 
                                test_normal_sample = 5000, seed = i)
    # Training the model
    validate = np.concatenate((normal_train_data, normal_val_data), axis=0)
    history = model.fit(normal_train_data,normal_train_data,
                        epochs = 500,
                        batch_size = 128,
                        validation_data = (validate,validate),
                        shuffle = True,
                        callbacks = [early_stopping], verbose = 0)
    reconstructions_a = model.predict(test_data)
    test_loss = tf.keras.losses.mae(reconstructions_a, test_data)
    
    prob_test = (test_loss-min(test_loss))/(max(test_loss)-min(test_loss))
    if best_strat == "AUC":
        fpr, tpr, thresholds = roc_curve(labels_test, prob_test)
        gmeans = sqrt(tpr * (1-fpr))
        ix = argmax(gmeans)
        threshold = thresholds[ix]
    elif best_strat == "F1":
        precision, recall, thresholds = precision_recall_curve(labels_test, prob_test)
        f1_scores = 2*recall*precision/(recall+precision)
        f1_scores = np.nan_to_num(f1_scores)
        threshold = thresholds[np.argmax(f1_scores)]
    # print(roc_auc_score(labels_test, prob_test, average=None))
    pred_test = [0 if prob_test[i] < threshold else 1 for i in range(len(prob_test))]
    print("Test Evaluation with Seed: " + str(i))
    print("AUC Score: " + str(roc_auc_score(labels_test, pred_test, average=None)))
    print(classification_report(labels_test ,pred_test,  labels=[0, 1], target_names=['benign', 'malicious']))
    print(confusion_matrix(labels_test ,pred_test))






