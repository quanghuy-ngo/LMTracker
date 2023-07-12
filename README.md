0. Download Comprehensive, Multi-Source Cyber-Security Events LANL data from: https://csr.lanl.gov/data/cyber1/
1. Preprocess redteam data:
  run the jupyter notebook script extract_redteam_auth.ipynb 
2. Contruct Graph from LANL data:
python3 graph_construction.py

3. Train the metapath2vec model:
python3 metapath2vec_torchgeo/metapath2vec.py

4. run autoecoder.ipynb to train from the attack path data generated from metapath2vec





Phase 1: replicate the paper [DONE]
+ all snapshot in phase1 folder
+ graph folder: graph_data_20220219110042
+ path embedding folder: model_20220219164849


Phase 2: expand the graph with the consideration of the authentication type
+ in progress








