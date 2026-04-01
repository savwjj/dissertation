# 将 CT 输入数据 变成类图 data 结构

#-*- coding:utf-8 -*-
import torch
import pandas as pd
from torch_geometric.data import Data
from torch_geometric.data import InMemoryDataset
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
from torch_geometric.nn import GraphConv,TopKPooling, ResGatedGraphConv, EdgePooling
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
import torch.nn.functional as F
import numpy as np
import os



# 构建图的边, 全连接
def create_edge_index_full_connection(length):
    b = [] # 存储源节点
    c = [] # 临时变量
    d = [] # 存储目标节点
    
    # 第一部分：构建源节点列表 b
    for i in range(length):
        a = [i] * (length - 1)
        b.extend(a)
    
    # 第二部分：构建目标节点列表 d
    for j in range(length):
        c = [k for k in range(length)]
        c.remove(j)
        d.extend(c)
    return torch.tensor([d, b], dtype=torch.long)


# 构建图的边,基于余弦相似度和阈值
def create_edge_indexCosine_similarity(node_features, threshold=0.0):
    features = node_features.numpy()
    # 对特征进行 L2 归一化
    features_norm = normalize(features, axis=1, norm='l2')
    # 计算余弦相似度矩阵
    similarity_matrix = cosine_similarity(features_norm)
    # 创建边连接
    edge_index = []
    for i in range(len(similarity_matrix)):
        for j in range(len(similarity_matrix[i])):
            if i != j and similarity_matrix[i][j] > threshold:
                edge_index.append([i, j])
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    return edge_index



class HNC_Dataset_BN(InMemoryDataset):
    def __init__(self, modality, root, datapath, data_set, transform=None, pre_transform=None):
        self.modality = modality
        self.data_set = data_set
        # self.label_m = label_m
        self.datapath = datapath
        self.dfTrain = pd.read_excel(os.path.join(self.datapath, "input_feas", f"{self.modality}_BN_train.xlsx"))
        self.dfTest = pd.read_excel(os.path.join(self.datapath, "input_feas", f"{self.modality}_BN_test.xlsx"))

        super(HNC_Dataset_BN, self).__init__(root, transform, pre_transform)
        # 吴京晶修改部分
        # self.data, self.slices = torch.load(self.processed_paths[0])
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)
    @property
    def raw_file_names(self):
        return []
    @property
    def processed_file_names(self):
        if self.data_set == 0:
            return [os.path.join(self.datapath, "dataset", f"HNC_{self.modality}_train_BN.dataset")]
        if self.data_set == 1:
            return [os.path.join(self.datapath, "dataset", f"HNC_{self.modality}_test_BN.dataset")]
    def download(self):
        pass
    def process(self):
        data_list = []
        # process by session_id
        # 按病人ID分组
        if self.data_set == 0:
            grouped = self.dfTrain.groupby('ID') # 所有相同ID的行归为一组
            # parameters_code_list =  sorted(self.dfTrain['parameters_code'].unique())
            parameters_code_list =  self.dfTrain['parameters_code'].unique()
            imageType_code_list =  self.dfTrain['imageType_code'].unique()
            column_names = self.dfTrain.columns.tolist()
            feal_list = column_names[21:]   # 提取第22列到最后一列（索引从0开始，21对应第22列）,即只提取影像组学特征
        if self.data_set == 1:
            grouped = self.dfTest.groupby('ID')
            parameters_code_list =  self.dfTest['parameters_code'].unique()
            imageType_code_list =  self.dfTest['imageType_code'].unique()
            column_names = self.dfTest.columns.tolist()
            feal_list = column_names[21:]   # 提取第22列到最后一列（索引从0开始，21对应第22列）,即只提取影像组学特征

        for ID, group in tqdm(grouped):
            sess_item_id = LabelEncoder().fit_transform(group.item_id)
            group = group.reset_index(drop=True)
            group['sess_item_id'] = sess_item_id
            length = len(group.item_id)
            
            node_features = group.loc[group.ID == ID, feal_list].values  # add clinical feature       # group.loc[行条件, 列条件]
            x = torch.tensor(node_features, dtype=torch.float)
            edge_index = create_edge_index_full_connection(length)  # 使用一个新函数创建 edge_index
            # edge_index = create_edge_indexCosine_similarity(node_features, threshold=0.5)   # 使用余弦连接 创建 edge_index
        
            # y = torch.tensor([group.OS_status.values[0]], dtype=torch.float)
            # y = torch.tensor([group.OS_5.values[0]], dtype=torch.float)
            OS_status = torch.tensor([group.OS_status.values[0]], dtype=torch.float)
            OS_time = torch.tensor([group.OS_time.values[0]], dtype=torch.float)
            PFS_status = torch.tensor([group.PFS_status.values[0]], dtype=torch.float)
            PFS_time = torch.tensor([group.PFS_time.values[0]], dtype=torch.float)
            RFS_status = torch.tensor([group.RFS_status.values[0]], dtype=torch.float)
            RFS_time = torch.tensor([group.RFS_time.values[0]], dtype=torch.float)
            MFS_status = torch.tensor([group.MFS_status.values[0]], dtype=torch.float)
            MFS_time = torch.tensor([group.MFS_time.values[0]], dtype=torch.float)
            Age = torch.tensor([group.Age.values[0]], dtype=torch.float)
            T_stage = torch.tensor([group.T_stage.values[0]], dtype=torch.float)
            N_stage = torch.tensor([group.N_stage.values[0]], dtype=torch.float)
            Site = torch.tensor([group.Site.values[0]], dtype=torch.float)
            # parameters_code = torch.tensor([group.parameters_code.values[0]], dtype=torch.float)
            # imageType_code = torch.tensor([group.imageType_code.values[0]], dtype=torch.float)

            data = Data(x=x, edge_index=edge_index, parameters_code=parameters_code_list, imageType_code=imageType_code_list,
                        OS_status=OS_status, OS_time=OS_time,
                        PFS_status=PFS_status, PFS_time=PFS_time,
                        RFS_status=RFS_status, RFS_time=RFS_time,
                        MFS_status=MFS_status, MFS_time=MFS_time,
                        Age=Age, T_stage=T_stage,
                        N_stage=N_stage, Site=Site
                        )
            # 吴京晶 新增：保存病人ID 
            data.patient_id = ID  # 添加这行，把ID保存到data对象中
            data_list.append(data)
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


        
class HNC_Dataset_addsr_BN(InMemoryDataset):
    def __init__(self, modality, root, datapath, data_set, transform=None, pre_transform=None):
        self.modality = modality
        self.data_set = data_set
        # self.label_m = label_m
        self.datapath = datapath
        self.dfTrain = pd.read_excel(os.path.join(self.datapath, "input_feas", f"{self.modality}_BN_train.xlsx"))
        self.dfTest = pd.read_excel(os.path.join(self.datapath, "input_feas", f"{self.modality}_BN_test.xlsx"))
        self.dfTestsr = pd.read_excel(os.path.join(self.datapath, "input_feas", f"{self.modality}_BN_testsr.xlsx"))

        super(HNC_Dataset_addsr_BN, self).__init__(root, transform, pre_transform)
        # 吴京晶修改部分
        # self.data, self.slices = torch.load(self.processed_paths[0])
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)
    @property
    def raw_file_names(self):
        return []
    @property
    def processed_file_names(self):
        if self.data_set == 0:
            return [os.path.join(self.datapath, "dataset", f"HNC_{self.modality}_train_BN.dataset")]
        if self.data_set == 1:
            return [os.path.join(self.datapath, "dataset", f"HNC_{self.modality}_test_BN.dataset")]
        if self.data_set == 2:
            return [os.path.join(self.datapath, "dataset", f"HNC_{self.modality}_testsr_BN.dataset")]
    def download(self):
        pass
    def process(self):
        data_list = []
        # process by session_id
        if self.data_set == 0:
            grouped = self.dfTrain.groupby('ID')
            parameters_code_list =  self.dfTrain['parameters_code'].unique()
            imageType_code_list =  self.dfTrain['imageType_code'].unique()
            column_names = self.dfTrain.columns.tolist()
            feal_list = column_names[21:]   # 提取第22列到最后一列（索引从0开始，21对应第22列）,即只提取影像组学特征
        if self.data_set == 1:
            grouped = self.dfTest.groupby('ID')
            parameters_code_list =  self.dfTest['parameters_code'].unique()
            imageType_code_list =  self.dfTest['imageType_code'].unique()
            column_names = self.dfTest.columns.tolist()
            feal_list = column_names[21:]   # 提取第22列到最后一列（索引从0开始，21对应第22列）,即只提取影像组学特征
        if self.data_set == 2:
            grouped = self.dfTestsr.groupby('ID')
            parameters_code_list =  self.dfTestsr['parameters_code'].unique()
            imageType_code_list =  self.dfTestsr['imageType_code'].unique()
            column_names = self.dfTestsr.columns.tolist()
            feal_list = column_names[21:]   # 提取第22列到最后一列（索引从0开始，21对应第22列）,即只提取影像组学特征

        for ID, group in tqdm(grouped):
            # 对item_id进行编码（将字符串转为数字）
            sess_item_id = LabelEncoder().fit_transform(group.item_id)
            group = group.reset_index(drop=True)
            group['sess_item_id'] = sess_item_id
            length = len(group.item_id)
            
            # 节点特征：每个item_id对应的影像组学特征
            node_features = group.loc[group.ID == ID, feal_list].values  # add clinical feature       # group.loc[行条件, 列条件]
            x = torch.tensor(node_features, dtype=torch.float)
            edge_index = create_edge_index_full_connection(length)  # 使用一个新函数创建 edge_index
            # edge_index = create_edge_indexCosine_similarity(node_features, threshold=0.5)   # 使用余弦连接 创建 edge_index
        
            # y = torch.tensor([group.OS_status.values[0]], dtype=torch.float)
            # y = torch.tensor([group.OS_5.values[0]], dtype=torch.float)
            # 生存数据：总生存、无进展生存、无复发生存、无转移生存
            OS_status = torch.tensor([group.OS_status.values[0]], dtype=torch.float)
            OS_time = torch.tensor([group.OS_time.values[0]], dtype=torch.float)
            PFS_status = torch.tensor([group.PFS_status.values[0]], dtype=torch.float)
            PFS_time = torch.tensor([group.PFS_time.values[0]], dtype=torch.float)
            RFS_status = torch.tensor([group.RFS_status.values[0]], dtype=torch.float)
            RFS_time = torch.tensor([group.RFS_time.values[0]], dtype=torch.float)
            MFS_status = torch.tensor([group.MFS_status.values[0]], dtype=torch.float)
            MFS_time = torch.tensor([group.MFS_time.values[0]], dtype=torch.float)
            # 临床特征：年龄、T分期、N分期、肿瘤部位
            Age = torch.tensor([group.Age.values[0]], dtype=torch.float)
            T_stage = torch.tensor([group.T_stage.values[0]], dtype=torch.float)
            N_stage = torch.tensor([group.N_stage.values[0]], dtype=torch.float)
            Site = torch.tensor([group.Site.values[0]], dtype=torch.float)
            # parameters_code = torch.tensor([group.parameters_code.values[0]], dtype=torch.float)
            # imageType_code = torch.tensor([group.imageType_code.values[0]], dtype=torch.float)

            data = Data(x=x, edge_index=edge_index, parameters_code=parameters_code_list, imageType_code=imageType_code_list,
                        OS_status=OS_status, OS_time=OS_time,
                        PFS_status=PFS_status, PFS_time=PFS_time,
                        RFS_status=RFS_status, RFS_time=RFS_time,
                        MFS_status=MFS_status, MFS_time=MFS_time,
                        Age=Age, T_stage=T_stage,
                        N_stage=N_stage, Site=Site
                        )
            # 吴京晶 新增：保存病人ID 
            data.patient_id = ID  # 添加这行，把ID保存到data对象中
            data_list.append(data)
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])



class HNC_Dataset_BS(InMemoryDataset):
    def __init__(self, modality, root, datapath, data_set, transform=None, pre_transform=None):
        self.modality = modality
        self.data_set = data_set
        # self.label_m = label_m
        self.datapath = datapath
        self.dfTrain = pd.read_excel(os.path.join(self.datapath, "input_feas", f"{self.modality}_BS_train.xlsx"))
        self.dfTest = pd.read_excel(os.path.join(self.datapath, "input_feas", f"{self.modality}_BS_test.xlsx"))

        super(HNC_Dataset_BS, self).__init__(root, transform, pre_transform)
        # 吴京晶修改部分
        #self.data, self.slices = torch.load(self.processed_paths[0])
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)
    @property
    def raw_file_names(self):
        return []
    @property
    def processed_file_names(self):
        if self.data_set == 0:
            return [os.path.join(self.datapath, "dataset", f"HNC_{self.modality}_train_BS.dataset")]
        if self.data_set == 1:
            return [os.path.join(self.datapath, "dataset", f"HNC_{self.modality}_test_BS.dataset")]
    def download(self):
        pass
    def process(self):
        data_list = []
        # process by session_id
        if self.data_set == 0:
            grouped = self.dfTrain.groupby('ID')
            parameters_code_list =  self.dfTrain['parameters_code'].unique()
            imageType_code_list =  self.dfTrain['imageType_code'].unique()
            column_names = self.dfTrain.columns.tolist()
            feal_list = column_names[21:]   # 提取第22列到最后一列（索引从0开始，21对应第22列）,即只提取影像组学特征
        if self.data_set == 1:
            grouped = self.dfTest.groupby('ID')
            parameters_code_list =  self.dfTest['parameters_code'].unique()
            imageType_code_list =  self.dfTest['imageType_code'].unique()
            column_names = self.dfTest.columns.tolist()
            feal_list = column_names[21:]   # 提取第22列到最后一列（索引从0开始，21对应第22列）,即只提取影像组学特征

        for ID, group in tqdm(grouped):
            sess_item_id = LabelEncoder().fit_transform(group.item_id)
            group = group.reset_index(drop=True)
            group['sess_item_id'] = sess_item_id
            length = len(group.item_id)
            
            node_features = group.loc[group.ID == ID, feal_list].values  # add clinical feature       # group.loc[行条件, 列条件]
            x = torch.tensor(node_features, dtype=torch.float)
            edge_index = create_edge_index_full_connection(length)  # 使用一个新函数创建 edge_index
            # edge_index = create_edge_indexCosine_similarity(node_features, threshold=0.5)   # 使用余弦连接 创建 edge_index
        
            # y = torch.tensor([group.OS_status.values[0]], dtype=torch.float)
            # y = torch.tensor([group.OS_5.values[0]], dtype=torch.float)
            OS_status = torch.tensor([group.OS_status.values[0]], dtype=torch.float)
            OS_time = torch.tensor([group.OS_time.values[0]], dtype=torch.float)
            PFS_status = torch.tensor([group.PFS_status.values[0]], dtype=torch.float)
            PFS_time = torch.tensor([group.PFS_time.values[0]], dtype=torch.float)
            RFS_status = torch.tensor([group.RFS_status.values[0]], dtype=torch.float)
            RFS_time = torch.tensor([group.RFS_time.values[0]], dtype=torch.float)
            MFS_status = torch.tensor([group.MFS_status.values[0]], dtype=torch.float)
            MFS_time = torch.tensor([group.MFS_time.values[0]], dtype=torch.float)
            Age = torch.tensor([group.Age.values[0]], dtype=torch.float)
            T_stage = torch.tensor([group.T_stage.values[0]], dtype=torch.float)
            N_stage = torch.tensor([group.N_stage.values[0]], dtype=torch.float)
            Site = torch.tensor([group.Site.values[0]], dtype=torch.float)
            # parameters_code = torch.tensor([group.parameters_code.values[0]], dtype=torch.float)
            # imageType_code = torch.tensor([group.imageType_code.values[0]], dtype=torch.float)

            data = Data(x=x, edge_index=edge_index, parameters_code=parameters_code_list, imageType_code=imageType_code_list,
                        OS_status=OS_status, OS_time=OS_time,
                        PFS_status=PFS_status, PFS_time=PFS_time,
                        RFS_status=RFS_status, RFS_time=RFS_time,
                        MFS_status=MFS_status, MFS_time=MFS_time,
                        Age=Age, T_stage=T_stage,
                        N_stage=N_stage, Site=Site
                        )
            # 吴京晶 新增：保存病人ID 
            data.patient_id = ID  # 添加这行，把ID保存到data对象中
            data_list.append(data)
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


        
class HNC_Dataset_addsr_BS(InMemoryDataset):
    def __init__(self, modality, root, datapath, data_set, transform=None, pre_transform=None):
        self.modality = modality
        self.data_set = data_set
        # self.label_m = label_m
        self.datapath = datapath
        self.dfTrain = pd.read_excel(os.path.join(self.datapath, "input_feas", f"{self.modality}_BS_train.xlsx"))
        self.dfTest = pd.read_excel(os.path.join(self.datapath, "input_feas", f"{self.modality}_BS_test.xlsx"))
        self.dfTestsr = pd.read_excel(os.path.join(self.datapath, "input_feas", f"{self.modality}_BS_testsr.xlsx"))

        super(HNC_Dataset_addsr_BS, self).__init__(root, transform, pre_transform)
        # 吴京晶修改部分
        # self.data, self.slices = torch.load(self.processed_paths[0])
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)
    @property
    def raw_file_names(self):
        return []
    @property
    def processed_file_names(self):
        if self.data_set == 0:
            return [os.path.join(self.datapath, "dataset", f"HNC_{self.modality}_train_BS.dataset")]
        if self.data_set == 1:
            return [os.path.join(self.datapath, "dataset", f"HNC_{self.modality}_test_BS.dataset")]
        if self.data_set == 2:
            return [os.path.join(self.datapath, "dataset", f"HNC_{self.modality}_testsr_BS.dataset")]
    def download(self):
        pass
    def process(self):
        data_list = []
        # process by session_id
        if self.data_set == 0:
            grouped = self.dfTrain.groupby('ID')
            parameters_code_list =  self.dfTrain['parameters_code'].unique()
            imageType_code_list =  self.dfTrain['imageType_code'].unique()
            column_names = self.dfTrain.columns.tolist()
            feal_list = column_names[21:]   # 提取第22列到最后一列（索引从0开始，21对应第22列）,即只提取影像组学特征
        if self.data_set == 1:
            grouped = self.dfTest.groupby('ID')
            parameters_code_list =  self.dfTest['parameters_code'].unique()
            imageType_code_list =  self.dfTest['imageType_code'].unique()
            column_names = self.dfTest.columns.tolist()
            feal_list = column_names[21:]   # 提取第22列到最后一列（索引从0开始，21对应第22列）,即只提取影像组学特征
        if self.data_set == 2:
            grouped = self.dfTestsr.groupby('ID')
            parameters_code_list =  self.dfTestsr['parameters_code'].unique()
            imageType_code_list =  self.dfTestsr['imageType_code'].unique()
            column_names = self.dfTestsr.columns.tolist()
            feal_list = column_names[21:]   # 提取第22列到最后一列（索引从0开始，21对应第22列）,即只提取影像组学特征

        for ID, group in tqdm(grouped):
            sess_item_id = LabelEncoder().fit_transform(group.item_id)
            group = group.reset_index(drop=True)
            group['sess_item_id'] = sess_item_id
            length = len(group.item_id)
            
            node_features = group.loc[group.ID == ID, feal_list].values  # add clinical feature       # group.loc[行条件, 列条件]
            x = torch.tensor(node_features, dtype=torch.float)
            edge_index = create_edge_index_full_connection(length)  # 使用一个新函数创建 edge_index
            # edge_index = create_edge_indexCosine_similarity(node_features, threshold=0.5)   # 使用余弦连接 创建 edge_index
        
            # y = torch.tensor([group.OS_status.values[0]], dtype=torch.float)
            # y = torch.tensor([group.OS_5.values[0]], dtype=torch.float)
            OS_status = torch.tensor([group.OS_status.values[0]], dtype=torch.float)
            OS_time = torch.tensor([group.OS_time.values[0]], dtype=torch.float)
            PFS_status = torch.tensor([group.PFS_status.values[0]], dtype=torch.float)
            PFS_time = torch.tensor([group.PFS_time.values[0]], dtype=torch.float)
            RFS_status = torch.tensor([group.RFS_status.values[0]], dtype=torch.float)
            RFS_time = torch.tensor([group.RFS_time.values[0]], dtype=torch.float)
            MFS_status = torch.tensor([group.MFS_status.values[0]], dtype=torch.float)
            MFS_time = torch.tensor([group.MFS_time.values[0]], dtype=torch.float)
            Age = torch.tensor([group.Age.values[0]], dtype=torch.float)
            T_stage = torch.tensor([group.T_stage.values[0]], dtype=torch.float)
            N_stage = torch.tensor([group.N_stage.values[0]], dtype=torch.float)
            Site = torch.tensor([group.Site.values[0]], dtype=torch.float)
            # parameters_code = torch.tensor([group.parameters_code.values[0]], dtype=torch.float)
            # imageType_code = torch.tensor([group.imageType_code.values[0]], dtype=torch.float)

            data = Data(x=x, edge_index=edge_index, parameters_code=parameters_code_list, imageType_code=imageType_code_list,
                        OS_status=OS_status, OS_time=OS_time,
                        PFS_status=PFS_status, PFS_time=PFS_time,
                        RFS_status=RFS_status, RFS_time=RFS_time,
                        MFS_status=MFS_status, MFS_time=MFS_time,
                        Age=Age, T_stage=T_stage,
                        N_stage=N_stage, Site=Site
                        )
            # 吴京晶 新增：保存病人ID 
            data.patient_id = ID  # 添加这行，把ID保存到data对象中
            data_list.append(data)
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

