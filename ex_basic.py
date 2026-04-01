# -*- coding: utf-8 -*-
# 第一行为吴京晶添加代码
# 进行 20 次五折交叉验证，将五次的训练集预测得分集成训练 cox 模型
# 统计 20 次验证集的 C 指数情况，根据 20 次验证集的 C 指数的均值观察 PCA 的 K 值情况
# 在 20 次五折循环得到的 100 个 model 中根据验证集 C 指数和训练集 C 指数挑选出较好的 5 个 model，然后进行 cox、coxcli、ave,查看性能情况
# 实验的数据集参数：all、somecolumns1、somecolumns1、PCA k=5,10,15……
# 早停策略使用loss 和最近的五次验证集 C 指数均值做早停

# -*- coding:utf-8 -*-
import os,sys   # os提供了许多与操作系统交互的函数的模块，可以用来处理文件和目录等操作; sys这个模块提供了许多有关Python运行时环境的变量和函数，用于操作Python解释器
# Set The WorkSpace Path
datapath = 'E:/bs/my_system/MPFGNN'
sys.path.append(datapath)  # 将指定的路径 /home/fugui/FRGCN/ 添加到 Python 解释器的 sys.path 列表中。sys.path 是一个包含模块搜索路径的列表，Python 解释器会按照列表中的顺序搜索模块文件。通过将特定路径添加到 sys.path 中，你可以告诉 Python 解释器在搜索模块时也包括这个路径

import numpy as np      # 用于科学计算的一个核心库，提供了高性能的多维数组对象以及用于处理这些数组的工具
import pandas as pd     # 数据处理工具，提供了数据结构和数据分析工具，特别适用于处理结构化数据
import torch            # PyTorch是一个开源的深度学习框架，提供了张量计算以及构建深度神经网络的功能
from torch_geometric.loader import DataLoader   # 导入了PyTorch Geometric中的DataLoader，用于加载图数据
from torch.utils.data import Subset
import heapq            # 堆队列算法模块，提供了堆队列算法的实现
import matplotlib.pyplot as plt     # 用于绘制图表和数据可视化的库，pyplot是Matplotlib的一个子模块，用于创建图形界面
# from HNC_data_lonely import HNC_Dataset_BN, HNC_Dataset_addsr_BN, HNC_Dataset_BS, HNC_Dataset_addsr_BS 
from HNC_data import HNC_Dataset_BN, HNC_Dataset_addsr_BN, HNC_Dataset_BS, HNC_Dataset_addsr_BS 
from model_ord_GCN import FinalModel # , Net4, Net5       # 导入了自定义的模型Net1, Net2, Net3 神经网络模型; here, design model
# from HNC_external_data import HNC_Datasetex
from utils import CoxLoss, cox_log_rank, draw_km, auc_cox, calculate_2year_metrics, stratified_pi_bootstrap_metrics, stratified_bca_bootstrap_metrics  # 导入了一些自定义的工具函数，比如用于计算准确率、损失函数等工具函数
from lifelines import CoxPHFitter       # Lifelines是一个用于生存分析的Python库，提供了生存分析相关的工具
from lifelines.utils import concordance_index
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.datasets import load_regression_dataset
from lifelines.utils import k_fold_cross_validation
from lifelines.statistics import logrank_test
from lifelines.plotting import add_at_risk_counts
import matplotlib.pyplot as plt     # 用于绘制图表和数据可视化的库，pyplot是Matplotlib的一个子模块，用于创建图形界面
from matplotlib.ticker import MultipleLocator
from sklearn import preprocessing       # Scikit-learn是一个机器学习库，提供了许多常用的机器学习算法和工具
from sklearn.linear_model import LassoCV    # 一个用于Lasso回归的交叉验证模型
from sklearn.model_selection import KFold   # 用于交叉验证中的K折划分数据集
from sklearn.model_selection import train_test_split
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset
import datetime
import time
import pickle
import concurrent.futures
import random
import seaborn as sns



# 计算训练集 损失
def train_loss(data_loader, data_length, model, optimizer, device):
    '''
    train model
    '''
    model.train()       # 将模型设置为训练模式。在 PyTorch 中，这通常用于将模型设置为训练模式，从而打开 dropout 和批量归一化等功能
    loss_all = 0        # 初始化一个变量loss_all来跟踪累积损失
    for data in data_loader:
        data = data.to(device)  # 将数据移到GPU/CPU
        optimizer.zero_grad()   # 清除所有优化后的梯度torch.Tensor。这是必要的，因为默认情况下会累积梯度（对于 RNN 和其他顺序数据很有用），因此需要在新的小批量开始时清除它们
        # output, out1, out2 = model(data)
        output = model(data)  # 前向传播，得到预测值


        censor = data.PFS_status.to(device)     # 将生存数据的审查信息移动到指定的设备
        survtime = data.PFS_time.to(device)     # 该行将生存时间数据移动到指定的设备


        loss = CoxLoss(survtime, censor, output, device)    # 使用自定义 Cox 损失函数计算损失，如何计算，公式？
        
        loss_all += data.num_graphs * loss.item()   # 累积当前小批量的损失
        loss.backward()     # 使用反向传播计算损失相对于模型参数的梯度
        optimizer.step()    # 根据计算的梯度更新模型的参数
    return loss_all / data_length


# 计算训测试集 损失
def test_loss(data_loader, data_length, model, device):
    '''
    evaluate model
    '''
    model.eval()
    loss_all = 0
    for data in data_loader:
        data = data.to(device)
        # output, out1, out2 = model(data)
        output = model(data)
        censor = data.PFS_status.to(device)
        survtime = data.PFS_time.to(device)
        loss = CoxLoss(survtime, censor, output, device)
        loss_all += data.num_graphs * loss.item()
    return loss_all / data_length


# 将所有批次的数据汇总后    计算的是整个数据集的综合性能指标
def test(data_loader, model, device):
    '''
    计算模型的预后性能指标：c-index, log-rank p value, auc/acc
    '''
    model.eval()            # 设置为评估模式。这通常用于关闭 dropout 和批量归一化等操作，因为在模型评估期间不需要它们
    predictions = []        # 初始化空列表以存储模型的预测、生存时间、事件指示器和标签
    time_test = []
    events_test = []

    with torch.no_grad():   # 该上下文管理器确保在其范围内不计算任何梯度。当不需要梯度时，这在模型评估期间非常有用
        for data in data_loader:
            data = data.to(device)
            # pred, out1, out2 = model(data)  # 使用模型对当前批次的数据进行预测，返回预测结果pred和其他两个输出out1和out2
            pred = model(data)  # 使用模型对当前批次的数据进行预测，返回预测结果pred和其他两个输出out1和out2
            pred[pred.isnan()] = 0      # 预测结果中存在NaN（不是一个数字），则将其替换为0
            pred = pred.detach().cpu().numpy()  # 预测结果从GPU移动到CPU，并将其从PyTorch张量转换为NumPy数组。detach()的作用是创建张量的副本，使其与计算图分离，这样在之后的计算中不会影响原始的计算图
            label = data.PFS_status.detach().cpu().numpy()
            predictions.append(pred)    # 将转换后的预测结果添加到predictions列表中
            time_test.append(data.PFS_time.detach().cpu().numpy())
            events_test.append(data.PFS_status.detach().cpu().numpy())

    predictions = np.hstack(predictions)        # 将预测、标签、生存时间和事件指示器的列表水平堆叠到单个 NumPy 数组中
    time_test = np.hstack(time_test)
    events_test = np.hstack(events_test)
    c_index = concordance_index(time_test, -predictions,
                                events_test)    # 计算预测结果与实际生存时间之间的一致性指数
    pvalue = cox_log_rank(predictions, events_test, time_test)  # 使用Cox回归模型的对数秩检验计算预测结果的p值
    rocmetrics = calculate_2year_metrics(predictions, events_test, time_test)   # 使用预测值和生存状态计算 ROC 相关指标，auc、sens、spec、acc

    return predictions, c_index, pvalue, rocmetrics


# 画 Loss 图
def get_Loss_pictures(round, now_fold, epoch, pic_path, train_Loss_list, valid_Loss_list):
    # 创建两个子图，分别展示训练和验证数据集的Cindex和损失值随着epoch的变化情况
    x = range(1, epoch + 2)

    # 创建主图  3行1列的网格布局
    fig, (ax, ax_zoom1, ax_zoom2) = plt.subplots(3, 1, figsize=(12, 18))    
    
    # 放大区域
    zoom_start = max(int(epoch * 0.5)-5, 1)
    zoom_end = min(int(epoch * 0.5)+5, len(x) - 1)
    # print(f"zoom_start - zoom_end：{zoom_start} - {zoom_end}")
    
    # Loss 图
    ax.plot(x, train_Loss_list, '.-', label="train_Loss", color='#1f77b4')  # 蓝色
    ax.plot(x, valid_Loss_list, '.-', label="valid_Loss", color='#ff7f0e')  # 橙色
    ax.set_title('Loss vs. Epochs', fontweight='bold')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Loss')
    ax.legend(loc='best')
    
    # Loss 图  放大区域
    ax_zoom1.plot(x[zoom_start:zoom_end], train_Loss_list[zoom_start:zoom_end], '.-', label="train_Loss", color='#1f77b4')  # 蓝色
    ax_zoom1.set_title(f'Zoomed One Loss of result (Epochs {zoom_start}-{zoom_end})', fontweight='bold')
    ax_zoom1.set_xlabel('Epochs')
    ax_zoom1.set_ylabel('Loss')
    ax_zoom1.legend(loc='best')
    
    # Loss 图  放大区域
    ax_zoom2.plot(x[zoom_start:zoom_end], valid_Loss_list[zoom_start:zoom_end], '.-', label="valid_Loss", color='#ff7f0e')  # 橙色
    ax_zoom2.set_title(f'Zoomed Two Loss of result (Epochs {zoom_start}-{zoom_end})', fontweight='bold')
    ax_zoom2.set_xlabel('Epochs')
    ax_zoom2.set_ylabel('Loss')
    ax_zoom2.legend(loc='best')
    
    # 调整子图间距
    plt.subplots_adjust(hspace=0.4)  # 增加上下子图间距

    # 保存主图
    plt.tight_layout()
    plt.savefig(os.path.join(pic_path, f'loss_picture_{round}_{now_fold}.png'))
    plt.close(fig)      # 关闭当前图形


# 画 top5 的 C 指数图
def get_top5_cindex_pictures(dtname, pic_path, train_Cindex_list, test_Cindex_list, testsr_Cindex_list):
    # 创建两个子图，分别展示训练和验证数据集的Cindex和损失值随着epoch的变化情况
    x = range(1, 6)

    # 创建主图  1行1列的网格布局
    fig, ax = plt.subplots(1, 1, figsize=(8, 10))

    # Cindex 图
    ax.plot(x, train_Cindex_list, '.-', label="train_Cindex", color='#1f77b4')    # 蓝色
    ax.plot(x, test_Cindex_list, '.-', label="test_Cindex", color='#2ca02c')     # 绿色
    ax.plot(x, testsr_Cindex_list, '.-', label="testsr_Cindex", color='#9467bd')   # 紫色
    ax.set_title('Cindex vs. Epochs', fontweight='bold')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Cindex')
    ax.legend(loc='best')

    # 保存主图
    plt.tight_layout()
    plt.savefig(os.path.join(pic_path, f'Cindex_picture_{dtname}.png'))
    plt.close(fig)      # 关闭当前图形


# top5 cox 模型
def top5_cox_model(dtname, pic_path, X_train, X_test, X_testsr, cli_train, cli_test, cli_testsr):
    train_coxdata = pd.concat([X_train, cli_train['PFS_time'], cli_train['PFS_status']], axis=1)
    test_coxdata = pd.concat([X_test, cli_test['PFS_time'], cli_test['PFS_status']], axis=1)
    testsr_coxdata = pd.concat([X_testsr, cli_testsr['PFS_time'], cli_testsr['PFS_status']], axis=1)

    cph = CoxPHFitter(penalizer=0.1)
    cph.fit(train_coxdata, 'PFS_time', event_col='PFS_status')
    # cph.print_summary()

    coxpred_train = cph.predict_partial_hazard(train_coxdata)
    coxpred_test = cph.predict_partial_hazard(test_coxdata)
    coxpred_testsr = cph.predict_partial_hazard(testsr_coxdata)

    cindex_train = cph.concordance_index_
    cindex_test = concordance_index(cli_test['PFS_time'], -coxpred_test, cli_test['PFS_status'])
    cindex_testsr = concordance_index(cli_testsr['PFS_time'], -coxpred_testsr, cli_testsr['PFS_status'])
    pvalue_train = cox_log_rank(coxpred_train, cli_train['PFS_status'], cli_train['PFS_time'])
    pvalue_test = cox_log_rank(coxpred_test, cli_test['PFS_status'], cli_test['PFS_time'])
    pvalue_testsr = cox_log_rank(coxpred_testsr, cli_testsr['PFS_status'], cli_testsr['PFS_time'])
    # cph.cindex_test = cindex_test
  
    draw_km(coxpred_train, cli_train['PFS_status'], cli_train['PFS_time'],
            os.path.join(pic_path, f'km_train_{dtname}'))
    draw_km(coxpred_test, cli_test['PFS_status'], cli_test['PFS_time'],
            os.path.join(pic_path, f'km_test_{dtname}'))
    draw_km(coxpred_testsr, cli_testsr['PFS_status'], cli_testsr['PFS_time'],
            os.path.join(pic_path, f'km_testsr_{dtname}'))

    # 计算两年 ROC 下的 auc、sens、spec、acc 指标
    train_rocmetrics = calculate_2year_metrics(coxpred_train, cli_train['PFS_status'], cli_train['PFS_time'])
    test_rocmetrics = calculate_2year_metrics(coxpred_test, cli_test['PFS_status'], cli_test['PFS_time'])
    testsr_rocmetrics = calculate_2year_metrics(coxpred_testsr, cli_testsr['PFS_status'], cli_testsr['PFS_time'])
    
    train_ci, _ = stratified_pi_bootstrap_metrics(coxpred_train, cli_train['PFS_status'], cli_train['PFS_time'])
    test_ci, _ = stratified_pi_bootstrap_metrics(coxpred_test, cli_test['PFS_status'], cli_test['PFS_time'])
    testsr_ci, _ = stratified_pi_bootstrap_metrics(coxpred_testsr, cli_testsr['PFS_status'], cli_testsr['PFS_time'])
    
    return {
        'coxpred_train': coxpred_train, 'coxpred_test': coxpred_test, 'coxpred_testsr': coxpred_testsr,
        'cindex_train': cindex_train, 'cindex_test': cindex_test, 'cindex_testsr': cindex_testsr,
        'pvalue_train': pvalue_train, 'pvalue_test': pvalue_test, 'pvalue_testsr': pvalue_testsr,
        
        'auc_train': train_rocmetrics['AUC'], 'auc_test': test_rocmetrics['AUC'], 'auc_testsr': testsr_rocmetrics['AUC'],
        'sens_train': train_rocmetrics['Sensitivity'], 'sens_test': test_rocmetrics['Sensitivity'], 'sens_testsr': testsr_rocmetrics['Sensitivity'],
        'spec_train': train_rocmetrics['Specificity'], 'spec_test': test_rocmetrics['Specificity'], 'spec_testsr': testsr_rocmetrics['Specificity'],
        'acc_train': train_rocmetrics['Accuracy'], 'acc_test': test_rocmetrics['Accuracy'], 'acc_testsr': testsr_rocmetrics['Accuracy'],
        'pre_train': train_rocmetrics['Precision'], 'pre_test': test_rocmetrics['Precision'], 'pre_testsr': testsr_rocmetrics['Precision'],
        'f1_train': train_rocmetrics['F1_score'], 'f1_test': test_rocmetrics['F1_score'], 'f1_testsr': testsr_rocmetrics['F1_score'],

        'ci_cindex_train': train_ci['Cindex'], 'ci_cindex_test': test_ci['Cindex'], 'ci_cindex_testsr': testsr_ci['Cindex'], 
        'ci_auc_train': train_ci['AUC'], 'ci_auc_test': test_ci['AUC'], 'ci_auc_testsr': testsr_ci['AUC'], 
        'ci_sens_train': train_ci['Sensitivity'], 'ci_sens_test': test_ci['Sensitivity'], 'ci_sens_testsr': testsr_ci['Sensitivity'], 
        'ci_spec_train': train_ci['Specificity'], 'ci_spec_test': test_ci['Specificity'], 'ci_spec_testsr': testsr_ci['Specificity'], 
        'ci_acc_train': train_ci['Accuracy'], 'ci_acc_test': test_ci['Accuracy'], 'ci_acc_testsr': testsr_ci['Accuracy'],
        'ci_pre_train': train_ci['Precision'], 'ci_pre_test': test_ci['Precision'], 'ci_pre_testsr': testsr_ci['Precision'], 
        'ci_f1_train': train_ci['F1_score'], 'ci_f1_test': test_ci['F1_score'], 'ci_f1_testsr': testsr_ci['F1_score']
    }


# top5 cox + cli 模型
def top5_cox_model_cli(dtname, pic_path, X_train, X_test, X_testsr, cli_train, cli_test, cli_testsr):
    train_coxdata = pd.concat([X_train, cli_train.iloc[:,[1,3,4,6]], cli_train['PFS_time'], cli_train['PFS_status']], axis=1)
    test_coxdata = pd.concat([X_test, cli_test.iloc[:,[1,3,4,6]], cli_test['PFS_time'], cli_test['PFS_status']], axis=1)
    testsr_coxdata = pd.concat([X_testsr, cli_testsr.iloc[:,[1,3,4,6]], cli_testsr['PFS_time'], cli_testsr['PFS_status']], axis=1)
    
    cph = CoxPHFitter(penalizer=0.1)
    cph.fit(train_coxdata, 'PFS_time', event_col='PFS_status')
    # cph.print_summary()
    
    coxpred_train = cph.predict_partial_hazard(train_coxdata)
    coxpred_test = cph.predict_partial_hazard(test_coxdata)
    coxpred_testsr = cph.predict_partial_hazard(testsr_coxdata)

    cindex_train = cph.concordance_index_
    cindex_test = concordance_index(cli_test['PFS_time'], -coxpred_test, cli_test['PFS_status'])
    cindex_testsr = concordance_index(cli_testsr['PFS_time'], -coxpred_testsr, cli_testsr['PFS_status'])
    pvalue_train = cox_log_rank(coxpred_train, cli_train['PFS_status'], cli_train['PFS_time'])
    pvalue_test = cox_log_rank(coxpred_test, cli_test['PFS_status'], cli_test['PFS_time'])
    pvalue_testsr = cox_log_rank(coxpred_testsr, cli_testsr['PFS_status'], cli_testsr['PFS_time'])
    
    draw_km(coxpred_train, cli_train['PFS_status'], cli_train['PFS_time'],
            os.path.join(pic_path, f'km_train_cli_{dtname}'))
    draw_km(coxpred_test, cli_test['PFS_status'], cli_test['PFS_time'],
            os.path.join(pic_path, f'km_test_cli_{dtname}'))
    draw_km(coxpred_testsr, cli_testsr['PFS_status'], cli_testsr['PFS_time'],
            os.path.join(pic_path, f'km_testsr_cli_{dtname}'))

    # 计算两年 ROC 下的 auc、sens、spec、acc 指标
    train_rocmetrics = calculate_2year_metrics(coxpred_train, cli_train['PFS_status'], cli_train['PFS_time'])
    test_rocmetrics = calculate_2year_metrics(coxpred_test, cli_test['PFS_status'], cli_test['PFS_time'])
    testsr_rocmetrics = calculate_2year_metrics(coxpred_testsr, cli_testsr['PFS_status'], cli_testsr['PFS_time'])
    
    train_ci, _ = stratified_pi_bootstrap_metrics(coxpred_train, cli_train['PFS_status'], cli_train['PFS_time'])
    test_ci, _ = stratified_pi_bootstrap_metrics(coxpred_test, cli_test['PFS_status'], cli_test['PFS_time'])
    testsr_ci, _ = stratified_pi_bootstrap_metrics(coxpred_testsr, cli_testsr['PFS_status'], cli_testsr['PFS_time'])
    
    return {
        'coxpred_train': coxpred_train, 'coxpred_test': coxpred_test, 'coxpred_testsr': coxpred_testsr,
        'cindex_train': cindex_train, 'cindex_test': cindex_test, 'cindex_testsr': cindex_testsr,
        'pvalue_train': pvalue_train, 'pvalue_test': pvalue_test, 'pvalue_testsr': pvalue_testsr,
        
        'auc_train': train_rocmetrics['AUC'], 'auc_test': test_rocmetrics['AUC'], 'auc_testsr': testsr_rocmetrics['AUC'],
        'sens_train': train_rocmetrics['Sensitivity'], 'sens_test': test_rocmetrics['Sensitivity'], 'sens_testsr': testsr_rocmetrics['Sensitivity'],
        'spec_train': train_rocmetrics['Specificity'], 'spec_test': test_rocmetrics['Specificity'], 'spec_testsr': testsr_rocmetrics['Specificity'],
        'acc_train': train_rocmetrics['Accuracy'], 'acc_test': test_rocmetrics['Accuracy'], 'acc_testsr': testsr_rocmetrics['Accuracy'],
        'pre_train': train_rocmetrics['Precision'], 'pre_test': test_rocmetrics['Precision'], 'pre_testsr': testsr_rocmetrics['Precision'],
        'f1_train': train_rocmetrics['F1_score'], 'f1_test': test_rocmetrics['F1_score'], 'f1_testsr': testsr_rocmetrics['F1_score'],

        'ci_cindex_train': train_ci['Cindex'], 'ci_cindex_test': test_ci['Cindex'], 'ci_cindex_testsr': testsr_ci['Cindex'], 
        'ci_auc_train': train_ci['AUC'], 'ci_auc_test': test_ci['AUC'], 'ci_auc_testsr': testsr_ci['AUC'], 
        'ci_sens_train': train_ci['Sensitivity'], 'ci_sens_test': test_ci['Sensitivity'], 'ci_sens_testsr': testsr_ci['Sensitivity'], 
        'ci_spec_train': train_ci['Specificity'], 'ci_spec_test': test_ci['Specificity'], 'ci_spec_testsr': testsr_ci['Specificity'], 
        'ci_acc_train': train_ci['Accuracy'], 'ci_acc_test': test_ci['Accuracy'], 'ci_acc_testsr': testsr_ci['Accuracy'],
        'ci_pre_train': train_ci['Precision'], 'ci_pre_test': test_ci['Precision'], 'ci_pre_testsr': testsr_ci['Precision'], 
        'ci_f1_train': train_ci['F1_score'], 'ci_f1_test': test_ci['F1_score'], 'ci_f1_testsr': testsr_ci['F1_score']
    }







# PCA 的数据类型名称 + allColumns
# my_datasetType_list = ["allColumns", "pcaColumns_10", "pcaColumns_20", "pcaColumns_30", "pcaColumns_40", "pcaColumns_50", 
#                         "pcaColumns_60", "pcaColumns_70", "pcaColumns_80", "pcaColumns_90", "pcaColumns_100"]

my_datasetType_list = ["pcaColumns_10"]

## 模型当前的模态 CT/PET, 模型的通道数 channel, 每一折模型循环的次数，筛选条件的阈值
my_modality = "CT"          # 模型当前的模态    CT/PT/CTPT
my_bintype = 'BN'           # my_bintype = 'BN'、'BN32'、'BN64'、'BN128'
my_channel = 128            # 模型通道数
num_epochs = 50             # 每一轮循环多少个 epoch
patience = 10               # 早停机制中允许的容忍轮数
loss_threshold = 0.001      # 设置 loss 的波动阈值,防止抖动
device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')     # 设置使用的cuda


# 加载临床文件
cli_train = pd.read_excel(os.path.join(datapath, "new_cli_feas", "HN_Clin_train.xlsx"))
cli_test = pd.read_excel(os.path.join(datapath, "new_cli_feas", "HN_Clin_test.xlsx"))
cli_testsr = pd.read_excel(os.path.join(datapath, "new_cli_feas", "HN_Clin_testsr.xlsx"))


# 加载 20次 训练验证划分的坐标 字典
with open(os.path.join(datapath, 'results_tv_div.pkl'), 'rb') as f:
    results_tv_div = pickle.load(f)
print("字典已从文件加载！")



# 结果存储路径
savepath = os.path.join(datapath, "final_results", f"result_GCN_{my_modality}_{my_channel}_{my_bintype}")
# 综合结果存储路径
synthesis_savepath = os.path.join(savepath, "synthesis_result")
# 所有详细结果存储路径
detailed_savepath = os.path.join(savepath, "detailed_result")

# 综合结果和图的存储路径
synthesis_saveresult_path = os.path.join(synthesis_savepath, "result")
synthesis_savepiccindex_path = os.path.join(synthesis_savepath, "cindex_pic")
synthesis_savepickm_path = os.path.join(synthesis_savepath, "km_pic")

# 创建综合结果和图的存储路径
if not os.path.exists(synthesis_saveresult_path):
    os.makedirs(synthesis_saveresult_path)
if not os.path.exists(synthesis_savepiccindex_path):
    os.makedirs(synthesis_savepiccindex_path)
if not os.path.exists(synthesis_savepickm_path):
    os.makedirs(synthesis_savepickm_path)



# 初始化保存所有 round 的结果的字典
# 吴京晶修改部分
# all_roundresult_dfdic = {key: pd.DataFrame() for key in my_datasetType_list}

# 最终最好的五个模型的集成结果
best5_info_dfdic = {key: pd.DataFrame() for key in my_datasetType_list}



best5_integrated_result_df = pd.DataFrame()
best5_integrated_addcli_result_df = pd.DataFrame()
best5_integrated_ave_result_df = pd.DataFrame()



# 获取当前日期
print(f"程序开始运行！此时时间为：{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")



#  主函数从此开始
if __name__ == '__main__': 
    for my_datasetType in my_datasetType_list:
        ## 加载数据集
        start_time_dataset = time.time()  # dataset 开始导入时间

        # 单离散化参数的情况 lonely 末尾 + “, bintype=my_bintype”
        train_dataset = HNC_Dataset_addsr_BN(my_modality, root = os.path.join(datapath, my_datasetType, "dataset"), datapath = os.path.join(datapath, my_datasetType), data_set=0) # 读入训练集
        test_dataset = HNC_Dataset_addsr_BN(my_modality, root = os.path.join(datapath, my_datasetType, "dataset"), datapath = os.path.join(datapath, my_datasetType), data_set=1) # 读入测试集
        testsr_dataset = HNC_Dataset_addsr_BN(my_modality, root = os.path.join(datapath, my_datasetType, "dataset"), datapath = os.path.join(datapath, my_datasetType), data_set=2) # 读入子区域测试集

        # train_dataset = HNC_Dataset_addsr_BN(my_modality, root = os.path.join(datapath, my_datasetType, "dataset"), datapath = os.path.join(datapath, my_datasetType), data_set=0, bintype=my_bintype) # 读入训练集
        # test_dataset = HNC_Dataset_addsr_BN(my_modality, root = os.path.join(datapath, my_datasetType, "dataset"), datapath = os.path.join(datapath, my_datasetType), data_set=1, bintype=my_bintype) # 读入测试集
        # testsr_dataset = HNC_Dataset_addsr_BN(my_modality, root = os.path.join(datapath, my_datasetType, "dataset"), datapath = os.path.join(datapath, my_datasetType), data_set=2, bintype=my_bintype) # 读入子区域测试集


        # num_workers = min(8, os.cpu_count())  # 根据CPU核心数调整num_workers
        # train_data_loader = DataLoader(train_dataset, batch_size=64, pin_memory=True, num_workers=num_workers)
        # test_data_loader = DataLoader(test_dataset, batch_size=64, pin_memory=True, num_workers=num_workers)
        # testsr_data_loader = DataLoader(testsr_dataset, batch_size=64, pin_memory=True, num_workers=num_workers)

        train_data_loader = DataLoader(train_dataset, batch_size=8, pin_memory=True)
        test_data_loader = DataLoader(test_dataset, batch_size=8, pin_memory=True)
        testsr_data_loader = DataLoader(testsr_dataset, batch_size=8, pin_memory=True)


        end_time_dataset = time.time()  # dataset 导入结束时间
        print(f"{my_datasetType} 数据集导入所需要的时间: {end_time_dataset - start_time_dataset:.6f} seconds")  # 数据集导入所需要的时间
        # 数据加载完毕
        
        # 设置数据集详细结果的存储路径
        nowdt_model_path = os.path.join(detailed_savepath, my_datasetType, "model")
        nowdt_picloss_path = os.path.join(detailed_savepath, my_datasetType, "loss_pic")

        
        # 创建当前数据集详细结果的存储路径
        if not os.path.exists(nowdt_model_path):
            os.makedirs(nowdt_model_path)
        if not os.path.exists(nowdt_picloss_path):
            os.makedirs(nowdt_picloss_path)


        # 循环 20 次五折交叉验证，0-100，间隔 5 为一个随机整数种子
        # 吴京晶修改部分，将循环全部注释掉
        # for round in range(0,100,5):
        #     start_time_round = time.time()      # round 开始时间
 
        #     for now_fold in range(5):
        #         start_time_nf = time.time()  # now_fold 开始时间
                
        #         # 进行每折实验
        #         # 当前 round 下的当前 now_fold 的训练验证的坐标
        #         index_train, index_valid = results_tv_div[(round, now_fold)]        # now_fold 五折交叉验证中的第几轮  kk 随机数种子      

        #         # 创建新的训练集和验证集 
        #         train_data = Subset(train_dataset, index_train)
        #         valid_data = Subset(train_dataset, index_valid)
        #         train_length = len(train_data)
        #         valid_length = len(valid_data)
                
        #         # 创建 DataLoader
        #         train_loader = DataLoader(train_data, batch_size=8, pin_memory=True)
        #         valid_loader = DataLoader(valid_data, batch_size=8, pin_memory=True)
                
        #         # 设置模型和优化器
        #         model = FinalModel(my_channel, train_dataset.num_features).to(device)     ## 初始化模型
        #         optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=0.001)  # Adam 优化器的作用是通过自适应调整学习率来高效地更新神经网络的权重

        #         # 初始化早停参数
        #         best_loss = float('inf')  # 初始化最佳验证损失
        #         best_cindex = float('-inf')  # 初始化最佳C指数
        #         patience_counter = 0  # 计数器
        #         cindex_history = []  # 用于记录C指数历史
                
        #         # 当前 round 的 now_fold 的 Loss
        #         train_Loss_list = []
        #         valid_Loss_list = []

        #         for epoch in range(num_epochs):
        #             loss_train = train_loss(train_loader, train_length, model, optimizer, device)
        #             loss_valid = test_loss(valid_loader, valid_length, model, device)
        #             train_Loss_list.append(loss_train)
        #             valid_Loss_list.append(loss_valid)
        #             print('Epoch: {:03d}, Loss_train: {:.5f}, Loss_valid: {:.5f}'.format(epoch, loss_train, loss_valid))

        #             # train_pred, train_cindex, train_pvalue, train_rocmetrics = test(train_loader, model, device)
        #             # 评估函数
        #             valid_pred, valid_cindex, valid_pvalue, valid_rocmetrics = test(valid_loader, model, device) 

        #             # 记录C指数历史
        #             cindex_history.append(valid_cindex)

        #             # 动态调整阈值（简单示例：使用历史C指数均值）
        #             valid_cindex_threshold = np.mean(cindex_history[-min(5, len(cindex_history)):]) if len(cindex_history) > 5 else 0.5

        #             # 早停逻辑
        #             if (loss_valid < best_loss - loss_threshold and valid_cindex > valid_cindex_threshold):  
        #                 best_loss = loss_valid
        #                 best_cindex = valid_cindex
        #                 patience_counter = 0  # 重置计数器
        #                 # 保存当前最佳模型状态
        #                 best_epoch = epoch
        #                 now_model_name = f'model_{round}_{now_fold}.pth'
        #                 best_model_path = os.path.join(nowdt_model_path, now_model_name)
        #                 torch.save(model.state_dict(), best_model_path)
        #                 print(f"Saved better model with loss: {best_loss:.5f}, C-index: {best_cindex:.5f}")
        #             else:  
        #                 patience_counter += 1

        #             # 打印当前的 patience_counter 值
        #             print(f"{my_datasetType} 的 {round} 轮的 {now_fold} 折的 Current patience_counter: {patience_counter}")

        #             # 早停条件
        #             if patience_counter >= patience:
        #                 print(f"now epoch is {epoch}! {my_datasetType} 的 {round} 轮的 Early stopping triggered. Stopping training.")
        #                 break

        #         # 画当前 round 的 now_fold 的 Loss 图
        #         # 损失函数绘制
        #         get_Loss_pictures(round, now_fold, epoch, nowdt_picloss_path, train_Loss_list, valid_Loss_list)

        #         # 在早停机制触发后，加载并使用保存的最佳模型
        #         model.load_state_dict(torch.load(best_model_path))
                
        #         # 计算原训练集、训练集、验证集、测试集、子区域测试集 性能评估
        #         trainor_pred, trainor_cindex, trainor_pvalue, trainor_rocmetrics = test(train_data_loader, model, device)
        #         train_pred, train_cindex, train_pvalue, train_rocmetrics = test(train_loader, model, device)
        #         valid_pred, valid_cindex, valid_pvalue, valid_rocmetrics = test(valid_loader, model, device) 
        #         test_pred, test_cindex, test_pvalue, test_rocmetrics = test(test_data_loader, model, device)
        #         testsr_pred, testsr_cindex, testsr_pvalue, testsr_rocmetrics = test(testsr_data_loader, model, device)

        #         roundresult_df = pd.DataFrame({'round': [round], 'now_fold': [now_fold], 'model_name': [now_model_name],
        #                                         'trainor_cindex': [trainor_cindex], 'train_cindex': [train_cindex], 'valid_cindex': [valid_cindex], 'test_cindex': [test_cindex], 'testsr_cindex': [testsr_cindex], 
        #                                         'trainor_pvalue': [trainor_pvalue], 'train_pvalue': [train_pvalue], 'valid_pvalue': [valid_pvalue], 'test_pvalue': [test_pvalue], 'testsr_pvalue': [testsr_pvalue],
        #                                         'trainor_auc': [trainor_rocmetrics['AUC']], 'train_auc': [train_rocmetrics['AUC']], 'valid_auc': [valid_rocmetrics['AUC']], 'test_auc': [test_rocmetrics['AUC']], 'testsr_auc': [testsr_rocmetrics['AUC']],
        #                                         'trainor_sens': [trainor_rocmetrics['Sensitivity']], 'train_sens': [train_rocmetrics['Sensitivity']], 'valid_sens': [valid_rocmetrics['Sensitivity']], 'test_sens': [test_rocmetrics['Sensitivity']], 'testsr_sens': [testsr_rocmetrics['Sensitivity']],
        #                                         'trainor_spec': [trainor_rocmetrics['Specificity']], 'train_spec': [train_rocmetrics['Specificity']], 'valid_spec': [valid_rocmetrics['Specificity']], 'test_spec': [test_rocmetrics['Specificity']], 'testsr_spec': [testsr_rocmetrics['Specificity']],
        #                                         'trainor_acc': [trainor_rocmetrics['Accuracy']], 'train_acc': [train_rocmetrics['Accuracy']], 'valid_acc': [valid_rocmetrics['Accuracy']], 'test_acc': [test_rocmetrics['Accuracy']], 'testsr_acc': [testsr_rocmetrics['Accuracy']],
        #                                         'trainor_pre': [trainor_rocmetrics['Precision']], 'train_pre': [train_rocmetrics['Precision']], 'valid_pre': [valid_rocmetrics['Precision']], 'test_pre': [test_rocmetrics['Precision']], 'testsr_pre': [testsr_rocmetrics['Precision']],
        #                                         'trainor_f1': [trainor_rocmetrics['F1_score']], 'train_f1': [train_rocmetrics['F1_score']], 'valid_f1': [valid_rocmetrics['F1_score']], 'test_f1': [test_rocmetrics['F1_score']], 'testsr_f1': [testsr_rocmetrics['F1_score']]
        #                                         })
        #         all_roundresult_dfdic[my_datasetType] = pd.concat([all_roundresult_dfdic[my_datasetType], roundresult_df], ignore_index=True)

        #         end_time_nf = time.time()  # now_fold 结束时间
        #         print(f"{my_datasetType} 的一折循环 (round={round} now_fold={now_fold}): {end_time_nf - start_time_nf:.6f} seconds")  # 输出每一折循环所需要的时间
          

        #     end_time_round = time.time()  # now_fold 结束时间
        #     print(f"{my_datasetType} 的(round={round}) 五折循环完毕！所需时间: {end_time_round - start_time_round:.6f} seconds")  # 输出一次五折循环所需要的时间

        
        # 获取当前数据集对应的 round*fold 的结果
        # 吴京晶修改部分
        # now_allroundresult_df = all_roundresult_dfdic[my_datasetType].copy()

        
        # 去除 train_cindex <= 0.5 或 trainor_cindex <=0.5 或 trainor_cindex >=0.64 的行
        # df_filtered = now_allroundresult_df[(now_allroundresult_df['train_cindex'] > 0.5) & (now_allroundresult_df['trainor_cindex'] > 0.5) & (now_allroundresult_df['trainor_cindex'] < 0.64)]
        # 吴京晶修改部分，下一行加注释
        # df_filtered = now_allroundresult_df[(now_allroundresult_df['train_cindex'] > 0.5) & (now_allroundresult_df['trainor_cindex'] > 0.5)]
        # 根据 valid_cindex 列降序排序  重置索引
        # 吴京晶修改部分，下一行加注释
        # df_sorted = df_filtered.sort_values(by=['valid_cindex', 'train_cindex'], ascending=[False, False]).reset_index(drop=True)

        # 计算 train_cindex 和 valid_cindex 差值
        # 吴京晶修改部分，下一行加注释
        # df_sorted['difference'] = (df_sorted['train_cindex'] - df_sorted['valid_cindex']).abs()
        # 吴京晶修改部分，下一行加注释
        # df_final = df_sorted[df_sorted['difference'] < 0.05]

        # 获取前五行的 model_name 列表
        # 吴京晶修改部分，下一行加注释
        # top5_model_names = df_final.head(5)['model_name'].tolist()
        # 吴京晶修改部分，将上一行注释掉，换成下面的top5_model_names
        top5_model_names = [
            'model_15_0.pth',  # 替换成你实际的最好的模型
            'model_5_0.pth',
            'model_50_3.pth',
            'model_60_0.pth',
            'model_75_3.pth'
        ]  
        # top 的预测值和 C 指数
        top5_train_pred_list = []
        top5_test_pred_list = []
        top5_testsr_pred_list = []

        top5_train_cindex_list = []
        top5_test_cindex_list = []
        top5_testsr_cindex_list = []

        # 吴京晶修改部分，添加model = FinalModel(my_channel, train_dataset.num_features).to(device)
        model = FinalModel(my_channel, train_dataset.num_features).to(device)

        for now_top5_model_name in top5_model_names:
            now_model_path = os.path.join(nowdt_model_path, now_top5_model_name)
            # 吴京晶修改部分
            # model.load_state_dict(torch.load(now_model_path))
            model.load_state_dict(torch.load(now_model_path, map_location='cpu'))
            train_pred, train_cindex, train_pvalue, train_rocmetrics = test(train_data_loader, model, device)
            test_pred, test_cindex, test_pvalue, test_rocmetrics = test(test_data_loader, model, device)
            testsr_pred, testsr_cindex, testsr_pvalue, testsr_rocmetrics = test(testsr_data_loader, model, device)

            top5_train_pred_list.append(train_pred)
            top5_test_pred_list.append(test_pred)
            top5_testsr_pred_list.append(testsr_pred)

            top5_train_cindex_list.append(train_cindex)
            top5_test_cindex_list.append(test_cindex)
            top5_testsr_cindex_list.append(testsr_cindex)

            now_top5_roundresult_df = pd.DataFrame({'model_name': [now_top5_model_name],
                                            'train_cindex': [train_cindex], 'test_cindex': [test_cindex], 'testsr_cindex': [testsr_cindex], 
                                            'train_pvalue': [train_pvalue], 'test_pvalue': [test_pvalue], 'testsr_pvalue': [testsr_pvalue],
                                            'train_auc': [train_rocmetrics['AUC']], 'test_auc': [test_rocmetrics['AUC']], 'testsr_auc': [testsr_rocmetrics['AUC']],
                                            'train_sens': [train_rocmetrics['Sensitivity']], 'test_sens': [test_rocmetrics['Sensitivity']], 'testsr_sens': [testsr_rocmetrics['Sensitivity']],
                                            'train_spec': [train_rocmetrics['Specificity']], 'test_spec': [test_rocmetrics['Specificity']], 'testsr_spec': [testsr_rocmetrics['Specificity']],
                                            'train_acc': [train_rocmetrics['Accuracy']], 'test_acc': [test_rocmetrics['Accuracy']], 'testsr_acc': [testsr_rocmetrics['Accuracy']],
                                            'train_pre': [train_rocmetrics['Precision']], 'test_pre': [test_rocmetrics['Precision']], 'testsr_pre': [testsr_rocmetrics['Precision']],
                                            'train_f1': [train_rocmetrics['F1_score']], 'test_f1': [test_rocmetrics['F1_score']], 'testsr_f1': [testsr_rocmetrics['F1_score']],
                                            })
            best5_info_dfdic[my_datasetType] = pd.concat([best5_info_dfdic[my_datasetType], now_top5_roundresult_df], ignore_index=True)

        # 画当前数据集的 top5 的 C 指数图
        get_top5_cindex_pictures(my_datasetType, synthesis_savepiccindex_path, top5_train_cindex_list, top5_test_cindex_list, top5_testsr_cindex_list)

        # 计算综合指标
        top5_train_pred_list = np.transpose(np.vstack(top5_train_pred_list))
        top5_test_pred_list = np.transpose(np.vstack(top5_test_pred_list))
        top5_testsr_pred_list = np.transpose(np.vstack(top5_testsr_pred_list))

        # 吴京晶修改部分，添加下面三行
        # 获取病人ID（假设数据集中有patient_id属性）
        train_patient_ids = [data.patient_id for data in train_dataset]
        test_patient_ids = [data.patient_id for data in test_dataset]
        testsr_patient_ids = [data.patient_id for data in testsr_dataset]

        #吴京晶修改部分，替换
        # 创建 DataFrame，列名为 'top0', 'top1', ..., 'top5'
        # top5_pred_fea_train = pd.DataFrame(top5_train_pred_list, columns=[f'top{rn}' for rn in range(5)])
        # top5_pred_fea_test = pd.DataFrame(top5_test_pred_list, columns=[f'top{rn}' for rn in range(5)])
        # top5_pred_fea_testsr = pd.DataFrame(top5_testsr_pred_list, columns=[f'top{rn}' for rn in range(5)])

        # 创建带ID的DataFrame
        top5_pred_fea_train = pd.DataFrame({
            'patient_id': train_patient_ids,
            **{f'top{rn}': top5_train_pred_list[:, rn] for rn in range(5)}
        })

        top5_pred_fea_test = pd.DataFrame({
            'patient_id': test_patient_ids,
            **{f'top{rn}': top5_test_pred_list[:, rn] for rn in range(5)}
        })

        top5_pred_fea_testsr = pd.DataFrame({
            'patient_id': testsr_patient_ids,
            **{f'top{rn}': top5_testsr_pred_list[:, rn] for rn in range(5)}
        })
        
        # 创建 ExcelWriter 对象
        with pd.ExcelWriter(os.path.join(synthesis_saveresult_path, f'{my_datasetType}_predictions_top5.xlsx'), engine='openpyxl') as writer:
            # 保存 top5_pred_fea_train 到一个工作表
            top5_pred_fea_train.to_excel(writer, sheet_name='train', index=False)
            # 保存 top5_pred_fea_test 到一个工作表
            top5_pred_fea_test.to_excel(writer, sheet_name='test', index=False)
            # 保存 top5_pred_fea_testsr 到一个工作表
            top5_pred_fea_testsr.to_excel(writer, sheet_name='testsr', index=False)

        # 吴京晶修改部分
        # top5_cox = top5_cox_model(my_datasetType, synthesis_savepickm_path, top5_pred_fea_train, top5_pred_fea_test, top5_pred_fea_testsr, cli_train, cli_test, cli_testsr)
        # 修改为：确保只传入数值列
        # 如果top5_pred_fea_train包含非数值列（如patient_id），只选择数值列
        numeric_cols = [col for col in top5_pred_fea_train.columns if col not in ['patient_id']]

        top5_cox = top5_cox_model(my_datasetType, synthesis_savepickm_path, 
                                top5_pred_fea_train[numeric_cols], 
                                top5_pred_fea_test[numeric_cols], 
                                top5_pred_fea_testsr[numeric_cols], 
                                cli_train, cli_test, cli_testsr)
        top5_coxcli = top5_cox_model_cli(my_datasetType, synthesis_savepickm_path, top5_cox['coxpred_train'], top5_cox['coxpred_test'], top5_cox['coxpred_testsr'], cli_train, cli_test, cli_testsr)

        top5_cox_df = pd.DataFrame({'dataset_type': [my_datasetType],
                                    'train_cindex': [f"{top5_cox['cindex_train']}\n({top5_cox['ci_cindex_train'][0]} - {top5_cox['ci_cindex_train'][1]})"], 
                                    'test_cindex': [f"{top5_cox['cindex_test']}\n({top5_cox['ci_cindex_test'][0]} - {top5_cox['ci_cindex_test'][1]})"], 
                                    'testsr_cindex': [f"{top5_cox['cindex_testsr']}\n({top5_cox['ci_cindex_testsr'][0]} - {top5_cox['ci_cindex_testsr'][1]})"], 
                                    'train_pvalue': [top5_cox['pvalue_train']], 'test_pvalue': [top5_cox['pvalue_test']], 'testsr_pvalue': [top5_cox['pvalue_testsr']], 
                                    'train_auc': [f"{top5_cox['auc_train']}\n({top5_cox['ci_auc_train'][0]} - {top5_cox['ci_auc_train'][1]})"], 
                                    'test_auc': [f"{top5_cox['auc_test']}\n({top5_cox['ci_auc_test'][0]} - {top5_cox['ci_auc_test'][1]})"], 
                                    'testsr_auc': [f"{top5_cox['auc_testsr']}\n({top5_cox['ci_auc_testsr'][0]} - {top5_cox['ci_auc_testsr'][1]})"], 
                                    'train_sens': [f"{top5_cox['sens_train']}\n({top5_cox['ci_sens_train'][0]} - {top5_cox['ci_sens_train'][1]})"], 
                                    'test_sens': [f"{top5_cox['sens_test']}\n({top5_cox['ci_sens_test'][0]} - {top5_cox['ci_sens_test'][1]})"], 
                                    'testsr_sens': [f"{top5_cox['sens_testsr']}\n({top5_cox['ci_sens_testsr'][0]} - {top5_cox['ci_sens_testsr'][1]})"], 
                                    'train_spec': [f"{top5_cox['spec_train']}\n({top5_cox['ci_spec_train'][0]} - {top5_cox['ci_spec_train'][1]})"], 
                                    'test_spec': [f"{top5_cox['spec_test']}\n({top5_cox['ci_spec_test'][0]} - {top5_cox['ci_spec_test'][1]})"], 
                                    'testsr_spec': [f"{top5_cox['spec_testsr']}\n({top5_cox['ci_spec_testsr'][0]} - {top5_cox['ci_spec_testsr'][1]})"], 
                                    'train_acc': [f"{top5_cox['acc_train']}\n({top5_cox['ci_acc_train'][0]} - {top5_cox['ci_acc_train'][1]})"], 
                                    'test_acc': [f"{top5_cox['acc_test']}\n({top5_cox['ci_acc_test'][0]} - {top5_cox['ci_acc_test'][1]})"], 
                                    'testsr_acc': [f"{top5_cox['acc_testsr']}\n({top5_cox['ci_acc_testsr'][0]} - {top5_cox['ci_acc_testsr'][1]})"], 
                                    'train_pre': [f"{top5_cox['pre_train']}\n({top5_cox['ci_pre_train'][0]} - {top5_cox['ci_pre_train'][1]})"], 
                                    'test_pre': [f"{top5_cox['pre_test']}\n({top5_cox['ci_pre_test'][0]} - {top5_cox['ci_pre_test'][1]})"], 
                                    'testsr_pre': [f"{top5_cox['pre_testsr']}\n({top5_cox['ci_pre_testsr'][0]} - {top5_cox['ci_pre_testsr'][1]})"], 
                                    'train_f1': [f"{top5_cox['f1_train']}\n({top5_cox['ci_f1_train'][0]} - {top5_cox['ci_f1_train'][1]})"], 
                                    'test_f1': [f"{top5_cox['f1_test']}\n({top5_cox['ci_f1_test'][0]} - {top5_cox['ci_f1_test'][1]})"], 
                                    'testsr_f1': [f"{top5_cox['f1_testsr']}\n({top5_cox['ci_f1_testsr'][0]} - {top5_cox['ci_f1_testsr'][1]})"]
                                    })
        
        top5_coxcli_df = pd.DataFrame({'dataset_type': [my_datasetType],
                                    'train_cindex': [f"{top5_coxcli['cindex_train']}\n({top5_coxcli['ci_cindex_train'][0]} - {top5_coxcli['ci_cindex_train'][1]})"], 
                                    'test_cindex': [f"{top5_coxcli['cindex_test']}\n({top5_coxcli['ci_cindex_test'][0]} - {top5_coxcli['ci_cindex_test'][1]})"], 
                                    'testsr_cindex': [f"{top5_coxcli['cindex_testsr']}\n({top5_coxcli['ci_cindex_testsr'][0]} - {top5_coxcli['ci_cindex_testsr'][1]})"], 
                                    'train_pvalue': [top5_coxcli['pvalue_train']], 'test_pvalue': [top5_coxcli['pvalue_test']], 'testsr_pvalue': [top5_coxcli['pvalue_testsr']], 
                                    'train_auc': [f"{top5_coxcli['auc_train']}\n({top5_coxcli['ci_auc_train'][0]} - {top5_coxcli['ci_auc_train'][1]})"], 
                                    'test_auc': [f"{top5_coxcli['auc_test']}\n({top5_coxcli['ci_auc_test'][0]} - {top5_coxcli['ci_auc_test'][1]})"], 
                                    'testsr_auc': [f"{top5_coxcli['auc_testsr']}\n({top5_coxcli['ci_auc_testsr'][0]} - {top5_coxcli['ci_auc_testsr'][1]})"], 
                                    'train_sens': [f"{top5_coxcli['sens_train']}\n({top5_coxcli['ci_sens_train'][0]} - {top5_coxcli['ci_sens_train'][1]})"], 
                                    'test_sens': [f"{top5_coxcli['sens_test']}\n({top5_coxcli['ci_sens_test'][0]} - {top5_coxcli['ci_sens_test'][1]})"], 
                                    'testsr_sens': [f"{top5_coxcli['sens_testsr']}\n({top5_coxcli['ci_sens_testsr'][0]} - {top5_coxcli['ci_sens_testsr'][1]})"], 
                                    'train_spec': [f"{top5_coxcli['spec_train']}\n({top5_coxcli['ci_spec_train'][0]} - {top5_coxcli['ci_spec_train'][1]})"], 
                                    'test_spec': [f"{top5_coxcli['spec_test']}\n({top5_coxcli['ci_spec_test'][0]} - {top5_coxcli['ci_spec_test'][1]})"], 
                                    'testsr_spec': [f"{top5_coxcli['spec_testsr']}\n({top5_coxcli['ci_spec_testsr'][0]} - {top5_coxcli['ci_spec_testsr'][1]})"], 
                                    'train_acc': [f"{top5_coxcli['acc_train']}\n({top5_coxcli['ci_acc_train'][0]} - {top5_coxcli['ci_acc_train'][1]})"], 
                                    'test_acc': [f"{top5_coxcli['acc_test']}\n({top5_coxcli['ci_acc_test'][0]} - {top5_coxcli['ci_acc_test'][1]})"], 
                                    'testsr_acc': [f"{top5_coxcli['acc_testsr']}\n({top5_coxcli['ci_acc_testsr'][0]} - {top5_coxcli['ci_acc_testsr'][1]})"], 
                                    'train_pre': [f"{top5_coxcli['pre_train']}\n({top5_coxcli['ci_pre_train'][0]} - {top5_coxcli['ci_pre_train'][1]})"], 
                                    'test_pre': [f"{top5_coxcli['pre_test']}\n({top5_coxcli['ci_pre_test'][0]} - {top5_coxcli['ci_pre_test'][1]})"], 
                                    'testsr_pre': [f"{top5_coxcli['pre_testsr']}\n({top5_coxcli['ci_pre_testsr'][0]} - {top5_coxcli['ci_pre_testsr'][1]})"], 
                                    'train_f1': [f"{top5_coxcli['f1_train']}\n({top5_coxcli['ci_f1_train'][0]} - {top5_coxcli['ci_f1_train'][1]})"], 
                                    'test_f1': [f"{top5_coxcli['f1_test']}\n({top5_coxcli['ci_f1_test'][0]} - {top5_coxcli['ci_f1_test'][1]})"], 
                                    'testsr_f1': [f"{top5_coxcli['f1_testsr']}\n({top5_coxcli['ci_f1_testsr'][0]} - {top5_coxcli['ci_f1_testsr'][1]})"]
                                    })
        
        top5_ave_df = pd.DataFrame({'dataset_type': [my_datasetType],
                                    'train_cindex': [np.mean(top5_train_cindex_list)], 
                                    'test_cindex': [np.mean(top5_test_cindex_list)], 
                                    'testsr_cindex': [np.mean(top5_testsr_cindex_list)]})
        
        best5_integrated_result_df = pd.concat([best5_integrated_result_df, top5_cox_df], ignore_index=True)
        best5_integrated_addcli_result_df = pd.concat([best5_integrated_addcli_result_df, top5_coxcli_df], ignore_index=True)
        best5_integrated_ave_result_df = pd.concat([best5_integrated_ave_result_df, top5_ave_df], ignore_index=True)

        ## 当前数据集处理完毕
        end_time_handledataset = time.time()  # dataset 开始导入时间
        print(f"{my_datasetType} 处理完毕！所需时间: {end_time_handledataset - start_time_dataset:.6f} seconds")  # 输出当前数据集的处理时间

    # 保存最好的五个模型的结果，包括cox、coxcli、ave 三种结果
    with pd.ExcelWriter(os.path.join(synthesis_saveresult_path, 'results_best5.xlsx')) as writer:
        best5_integrated_result_df.to_excel(writer, sheet_name = 'results_cox', index=False)
        best5_integrated_addcli_result_df.to_excel(writer, sheet_name = 'results_coxcli', index=False)
        best5_integrated_ave_result_df.to_excel(writer, sheet_name = 'results_ave', index=False)

    # 保存所有数据集的 最好的五个模型的详细信息
    with pd.ExcelWriter(os.path.join(synthesis_saveresult_path, "results_best5_detailed.xlsx"), engine='openpyxl') as writer:
        for my_datasetType in my_datasetType_list:
            df = best5_info_dfdic[my_datasetType]
            df = df.sort_values(by=['testsr_cindex'], ascending=[False]).reset_index(drop=True)
            df.to_excel(writer, sheet_name=my_datasetType, index=False)
        

    # 保存所有round的 DataFrame 到对应的 Excel 文件
    # with pd.ExcelWriter(os.path.join(synthesis_saveresult_path, "results_allround.xlsx"), engine='openpyxl') as writer:
    #     for my_datasetType in my_datasetType_list:
    #         df = all_roundresult_dfdic[my_datasetType]
    #         df = df.sort_values(by=['round', 'now_fold'], ascending=[True, True]).reset_index(drop=True)
    #         df.to_excel(writer, sheet_name=my_datasetType, index=False)

# 获取当前日期
# 吴京晶新增
# ==================== 最简单的查询 ====================
print("\n" + "="*60)
print("程序运行完成！")
print("="*60)

# Excel文件路径
excel_file = os.path.join(synthesis_saveresult_path, 'pcaColumns_10_predictions_top5.xlsx')

if os.path.exists(excel_file):
    print(f"\n预测文件已生成: {excel_file}")
    print("\n你可以：")
    print("1. 直接打开Excel文件查看")
    print("2. 在Python中查询特定病人")
    
    query = input("\n是否查询特定病人？(y/n): ").strip().lower()
    if query == 'y':
        patient_id = input("请输入病人ID: ").strip()
        
        # 读取并查找
        train_df = pd.read_excel(excel_file, sheet_name='train')
        test_df = pd.read_excel(excel_file, sheet_name='test')
        testsr_df = pd.read_excel(excel_file, sheet_name='testsr')
        
        # 合并所有数据
        all_df = pd.concat([train_df, test_df, testsr_df], ignore_index=True)
        
        # 查找病人
        result = all_df[all_df['patient_id'] == patient_id]
        
        if len(result) > 0:
            row = result.iloc[0]
            print(f"\n病人 {patient_id} 的预测得分:")
            print(f"  top0: {row['top0']:.6f}")
            print(f"  top1: {row['top1']:.6f}")
            print(f"  top2: {row['top2']:.6f}")
            print(f"  top3: {row['top3']:.6f}")
            print(f"  top4: {row['top4']:.6f}")
            print(f"  平均值: {(row['top0']+row['top1']+row['top2']+row['top3']+row['top4'])/5:.6f}")
        else:
            print(f"未找到病人ID: {patient_id}")
else:
    print(f"警告：预测文件不存在 - {excel_file}")

print(f"程序结束运行！此时时间为：{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")






