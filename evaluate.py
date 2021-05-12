import os
import numpy as np
import pandas as pd
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import pickle
import Graph4DIV
import torch
from data_process import *

MAXDOC = 50
REL_LEN = 18


def adjust_graph(A, rel_score_list, degree_tensor, selected_doc_id):
    '''
    adjust adjancent matrix A during the testing process, set the selected doc degree = 0
    :param rel_score_list: initial relevance of the document
    :param degree_tensor: degree tensor of each document
    :return: adjacent matrix A, degree tensor
    '''
    ''' connect selected document to the query node '''
    A[0, selected_doc_id+1, 0] = rel_score_list[selected_doc_id]
    A[0, 0, selected_doc_id+1] = rel_score_list[selected_doc_id]
    ''' remove edges between selected document and candidates '''
    A[0, selected_doc_id+1, 1:] = torch.tensor([0.0]*50).float()
    A[0, 1:, selected_doc_id+1] = torch.tensor([0.0]*50).float()
    ''' set the degree of selected document '''
    degree_tensor[0, selected_doc_id] = torch.tensor(0.0)
    return A, degree_tensor


def get_metric_nDCG_random(model, test_tuple, div_q, qid):
    '''
    get the alpha-nDCG for the input query, the input document list are randomly shuffled. 
    :param test_tuple: the features of the test query qid, test_turple = (feature, index, rel_feat, rel_score, A, degree)
    :param div_q: the div_query object of the test query qid
    :param qid: the id for the test query
    :return: the alpha-nDCG for the test query
    '''
    metric = 0
    end = Max_doc_num = len(div_q.best_docs_rank)
    current_docs_rank = []
    if not test_tuple:
        return 0 
    else:
        feature = test_tuple[0]
        index = test_tuple[1]
        rel_feat_tensor = torch.tensor(test_tuple[2]).float()
        rel_score_list = test_tuple[3]
        A = test_tuple[4]
        degree_tensor = test_tuple[5]

        A.requires_grad = False
        degree_tensor.requires_grad = False
        rel_feat_tensor.requires_grad = False
        lt = len(rel_score_list)
        if lt < MAXDOC:
            rel_score_list.extend([0.0]*(MAXDOC-lt))
        rel_score = torch.tensor(rel_score_list).float()
        
        A = A.reshape(1, A.shape[0], A.shape[1])
        feature = feature.reshape(1, feature.shape[0], feature.shape[1])
        rel_feat_tensor = rel_feat_tensor.reshape(1, rel_feat_tensor.shape[0], rel_feat_tensor.shape[1])
        degree_tensor = degree_tensor.reshape(1, degree_tensor.shape[0], degree_tensor.shape[1])
        
        if th.cuda.is_available():
            A = A.cuda()
            feature = feature.cuda()
            rel_feat_tensor = rel_feat_tensor.cuda()
            degree_tensor = degree_tensor.cuda()
        
        while len(current_docs_rank)<Max_doc_num:
            outputs = model(A, feature, rel_feat_tensor, degree_tensor)
            out = outputs.cpu().detach().numpy()
            result = np.argsort(-out[:end])

            for i in range(len(result)):
                if result[i] < Max_doc_num and index[result[i]] not in current_docs_rank:
                    current_docs_rank.append(index[result[i]])
                    adjust_index = result[i]
                    break
            A, degree_tensor = adjust_graph(A, rel_score, degree_tensor, adjust_index)

        if len(current_docs_rank)>0:
            new_docs_rank = [div_q.doc_list[i] for i in current_docs_rank]
            metric = div_q.get_test_alpha_nDCG(new_docs_rank)
    return metric


def get_docs_rank(model, test_tuple, div_q, qid):
    '''
    get the document ranking for the input query, the input document list are randomly shuffled. 
    :param test_tuple: the features of the test query qid,  test_turple = (feature, index, rel_feat, rel_score, A, degree)
    :param div_q: the div_query object of the test query qid
    :param qid: the id for the test query
    :return: the diversity ranking for the test query
    '''
    docs_rank = []
    current_docs_rank = []
    end = Max_doc_num = len(div_q.best_docs_rank)
    if not test_tuple:
        return []
    else:
        feature = test_tuple[0]
        index = test_tuple[1]
        rel_feat_tensor = torch.tensor(test_tuple[2]).float()
        rel_score_list = test_tuple[3]
        A = test_tuple[4]
        degree_tensor = test_tuple[5]
        
        A.requires_grad = False
        degree_tensor.requires_grad = False
        rel_feat_tensor.requires_grad = False
        lt = len(rel_score_list)
        if lt<MAXDOC:
            rel_score_list.extend([0.0]*(MAXDOC-lt))
        rel_score = torch.tensor(rel_score_list).float()
        
        A = A.reshape(1, A.shape[0], A.shape[1])
        feature = feature.reshape(1, feature.shape[0], feature.shape[1])
        rel_feat_tensor = rel_feat_tensor.reshape(1, rel_feat_tensor.shape[0], rel_feat_tensor.shape[1])
        degree_tensor = degree_tensor.reshape(1, degree_tensor.shape[0], degree_tensor.shape[1])
        if th.cuda.is_available():
            A = A.cuda()
            feature = feature.cuda()
            rel_feat_tensor = rel_feat_tensor.cuda()
            degree_tensor = degree_tensor.cuda()
        while len(current_docs_rank) < Max_doc_num:
            outputs = model(A, feature, rel_feat_tensor, degree_tensor)
            out = outputs.cpu().detach().numpy()
            result = np.argsort(-out[:end])
            for i in range(len(result)):
                if result[i] < Max_doc_num and index[result[i]] not in current_docs_rank:
                    current_docs_rank.append(index[result[i]])
                    adjust_index = result[i]
                    break
            A, degree_tensor = adjust_graph(A, rel_score, degree_tensor, adjust_index)
        if len(current_docs_rank)>0:
            new_docs_rank = [div_q.doc_list[i] for i in current_docs_rank]
            return new_docs_rank
    return []


def get_metrics_20(csv_file_path):
    '''
    get the final metrics from the output csv file
    :param csv_file_path: the output file of the ndeval test
    '''
    all_qids = range(1, 201)
    del_index = [94, 99]
    all_qids = np.delete(all_qids, del_index)
    qids = [str(i) for i in all_qids]
    df = pd.read_csv(csv_file_path)
    alpha_nDCG_20 = df.loc[df['topic'].isin(qids)]['alpha-nDCG@20'].mean()
    ERR_IA_20 = df.loc[df['topic'].isin(qids)]['ERR-IA@20'].mean()
    NRBP_20 = df.loc[df['topic'].isin(qids)]['NRBP'].mean()
    S_rec_20 = df.loc[df['topic'].isin(qids)]['strec@20'].mean()
    return alpha_nDCG_20, ERR_IA_20, NRBP_20, S_rec_20


def get_global_fullset_metric(best_model_list, test_qids_list, dump_dir, EMB_TYPE, EMB_LEN):
    '''
    get the final metrics for the five fold best models.
    :param best_model_list: the best models for the five corresponding folds.
    :param test_qids_list: the corresponding test qids for five folds.
    :param dump_dir: the document ranking output dir.
    '''
    output_file = dump_dir + 'run'
    fout = open(output_file, 'w')
    all_models = best_model_list

    test_graph_path = 'data/gcn_dataset/' + str(EMB_TYPE) + '_test_graph.data'
    test_graph_dict = pickle.load(open(test_graph_path, 'rb'))
    qd = pickle.load(open('data/gcn_dataset/div_query.data', 'rb'))
    std_metric = []

    ''' get the metrics for five folds '''
    for i in range(len(all_models)):
        test_qids = test_qids_list[i]
        model_file = all_models[i]
        model = Graph4DIV.Graph4Div(node_feature_dim = EMB_LEN,  hidden_dim = [EMB_LEN, EMB_LEN],  output_dim = EMB_LEN)
        model.load_state_dict(th.load(model_file))

        model.eval()
        if th.cuda.is_available():
            model = model.cuda()

        ''' ndeval test '''
        for qid in test_qids:
            docs_rank = get_docs_rank(model, test_graph_dict[str(qid)], qd[str(qid)], str(qid))
            if len(docs_rank)>0:
                for index in range(len(docs_rank)):
                    content = str(qid) + ' Q0 ' + str(docs_rank[index]) + ' ' + str(index+1) + ' -4.04239 indri\n'
                    fout.write(content)
    fout.close()
    csv_path = dump_dir+'result.csv'
    command = './eval/ndeval ./eval/2009-2012.diversity.ndeval.qrels ' + output_file + ' >' + str(csv_path)
    os.system(command)
    
    alpha_nDCG_20, ERR_IA_20, NRBP_20, S_rec_20 = get_metrics_20(csv_path)
    print('alpha_nDCG@20_std = {}, NRBP_20 = {}, ERR_IA_20 = {}, S_rec_20 = {}'.format(alpha_nDCG_20, NRBP_20, ERR_IA_20, S_rec_20))


def test_std_models(model_dir, EMB_TYPE, EMB_LEN):
    '''
    get the metrics for the stand models
    :param model_dir: the dir of the stand models
    '''
    all_qids = np.load('./data/gcn_dataset/all_qids.npy')
    output_file = model_dir + 'run'
    fout = open(output_file, 'w')
    all_models = os.listdir(model_dir)
    all_models.sort()
    all_models = [model_dir + str(all_models[i]) for i in range(len(all_models))]
    test_graph_path = 'data/gcn_dataset/' + str(EMB_TYPE) + '_test_graph.data'
    test_graph_dict = pickle.load(open(test_graph_path, 'rb'))
    qd = pickle.load(open('data/gcn_dataset/div_query.data', 'rb'))
    std_metric = []
    metrics = []
    test_qids_list = []
    for train_ids,  test_ids in KFold(5).split(all_qids):
        test_ids.sort()
        test_qids = [str(all_qids[i]) for i in test_ids]
        test_qids_list.append(test_qids)
    ''' test the corresponding five models for the five folds '''
    for i in range(5):
        test_qids = test_qids_list[i]
        model_file = all_models[i]
        model = Graph4DIV.Graph4Div(node_feature_dim = EMB_LEN,  hidden_dim = [EMB_LEN, EMB_LEN],  output_dim = EMB_LEN)
        model.load_state_dict(th.load(model_file))
        model.eval()
        if th.cuda.is_available():
            model = model.cuda()

        ''' ndeval test ''' 
        for qid in test_qids:
            docs_rank = get_docs_rank(model, test_graph_dict[str(qid)], qd[str(qid)], str(qid))
            if len(docs_rank)>0:
                for index in range(len(docs_rank)):
                    content = str(qid) + ' Q0 '+ str(docs_rank[index]) +' '+str(index+1) + ' -4.04239 indri\n'
                    fout.write(content)
    fout.close()
    csv_path = model_dir + 'result.csv'
    command = './eval/ndeval ./eval/2009-2012.diversity.ndeval.qrels ' + output_file + ' >' + str(csv_path)
    os.system(command)
    alpha_nDCG_20, ERR_IA_20, NRBP_20, S_rec_20 = get_metrics_20(csv_path)
    print('alpha_nDCG@20_std = {}, NRBP_20 = {}, ERR_IA_20 = {}, S_rec_20 = {}'.format(alpha_nDCG_20, NRBP_20, ERR_IA_20, S_rec_20))


def evaluate_accuracy(y_pred, y_label):
    num = len(y_pred)
    all_acc = 0.0
    count = 0
    for i in range(num):
        pred = (y_pred[i] > 0.5).astype(int)
        label = y_label[i]
        acc = 1 if pred == label else 0
        all_acc += acc
        count += 1
    return all_acc / count
