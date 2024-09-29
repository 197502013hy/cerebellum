
import os
import sys
import numpy as np
import pandas as pd
import pcntoolkit as ptk
import time
from warnings import filterwarnings
filterwarnings('ignore')
os.system('10fold')
for i in range(0,28):
    NUM = i
    print(NUM)
    # path = "/path/to/folder"
    DATA_PATH = 'G:/suit_seg/Nmodel/10fold/{}'.format(str(NUM))
    # new_folder = "new_folder"
    # new_folder_path = os.path.join(path, new_folder)
    # if not os.path.exists(new_folder_path):
    #     os.makedirs(new_folder_path)
    if not os.path.exists(DATA_PATH):
        print('----')
        os.makedirs(DATA_PATH)
        print('666')
    os.chdir(DATA_PATH)

    # RUN SIEMENS
    Y_my = pd.read_csv('G:/suit_seg/Nmodel/NC_VBM.csv', index_col = 0).iloc[:65, NUM]
    X_my = pd.read_csv('G:/suit_seg/Nmodel/NC_info_covariates.csv', index_col = 0).iloc[:66, :]

    batch_effects_tr = X_my['cohort'].to_numpy(dtype=int)
    uni = np.unique(batch_effects_tr)
    for index, i in enumerate(uni):
        np.place(batch_effects_tr, batch_effects_tr==i, index)
    n_values = np.max(batch_effects_tr) + 1
    bf_tr = np.eye(n_values)[batch_effects_tr]

    X_my[[i for i in range(len(uni))]] = bf_tr
    X_my[['age', 'sex', 'TIV'] + [i for i in range(len(uni))]].to_csv('X_train.csv',
        sep = ' ',
        header = False,
        index = False)
    Y_my.to_csv('Y_train.csv',
        sep = ' ',
        header = False,
        index = False)
    print('{} Loaded Train Data'.format(time.asctime( time.localtime(time.time()))))

    # create test data
    # RUN SIEMENS
    Y_my = pd.read_csv('G:/suit_seg/Nmodel/NC_VBM.csv', index_col = 0).iloc[:65, NUM]
    X_my = pd.read_csv('G:/suit_seg/Nmodel/NC_info_covariates.csv', index_col = 0).iloc[:66, :]
    # RUN GE
    # Y_my = pd.read_csv('/n02dat01/users/xwu/other/lchai/xwu/data/n00_GE_Cere_3mm_mask.csv', index_col = 0)
    # X_my = pd.read_csv('/n02dat01/users/xwu/other/lchai/xwu/data/n00_Ge_sites&cov_info.csv', index_col = 0)

    batch_effects_ts = X_my['cohort'].to_numpy(dtype=int)
    uni = np.unique(batch_effects_ts)
    for index, i in enumerate(uni):
        np.place(batch_effects_ts, batch_effects_ts==i, index)
    n_values = np.max(batch_effects_ts) + 1
    bf_ts = np.eye(n_values)[batch_effects_ts]

    X_my[[i for i in range(len(uni))]] = bf_ts
    X_my[['age', 'sex', 'TIV']+ [i for i in range(len(uni))]].to_csv('X_test.csv',
        sep = ' ',
        header = False,
        index = False)
    Y_my.to_csv('Y_test.csv',
        sep = ' ',
        header = False,
        index = False)
    print('{} Loaded Test Data'.format(time.asctime( time.localtime(time.time()))))


    # GPR: estimate forward model
    processing_dir = DATA_PATH

    respfile = os.path.join(processing_dir, 'Y_train.csv')       # measurements  (eg cortical thickness) of the training samples (columns: the various features/ROIs, rows: observations or subjects)
    covfile = os.path.join(processing_dir, 'X_train.csv')        # covariates (eg age) the training samples (columns: covariates, rows: observations or subjects)

    testrespfile_path = os.path.join(processing_dir, 'Y_test.csv')       # measurements  for the testing samples
    testcovfile_path = os.path.join(processing_dir, 'X_test.csv')        # covariate file for the testing samples

    output_path = os.path.join(processing_dir, 'Models_',str(NUM),'/')    #  output path, where the models will be written
    log_dir = os.path.join(processing_dir, 'log_',str(NUM),'/')           #
    if not os.path.isdir(output_path):
       os.mkdir(output_path)
    if not os.path.isdir(log_dir):
       os.mkdir(log_dir)

    outputsuffix = '_GPR'      # a string to name the output files, of use only to you, so adapt it for your needs.

    ptk.normative.estimate(covfile = covfile,
                respfile = respfile,
                alg = 'gpr',
                log_path = log_dir,
                output_path = output_path,
                cvfolds = 10,
                outputsuffix = outputsuffix,
                savemodel = True)
