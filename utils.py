# Numerical / Array
import lifelines
from lifelines.utils import concordance_index
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.datasets import load_regression_dataset
from lifelines.utils import k_fold_cross_validation
from lifelines.statistics import logrank_test
from sklearn.linear_model import LassoCV
from lifelines.plotting import add_at_risk_counts
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import average_precision_score, auc, f1_score, roc_curve, roc_auc_score, confusion_matrix
from sklearn.preprocessing import LabelBinarizer
# from sklearn.utils.testing import ignore_warnings
# from scipy import interp
from scipy.interpolate import interp1d
# mpl.rcParams['axes.linewidth'] = 3 #set the value globally

# Torch
import torch
import torch.nn as nn
from torch.nn import init, Parameter
from torch.utils.data._utils.collate import *
from torch.utils.data.dataloader import default_collate
import torch_geometric
from torch_geometric.data import Batch
from matplotlib.ticker import MultipleLocator
from sklearn.utils import resample
from scikits.bootstrap import ci  # 需要安装：pip install scikits.bootstrap



def accuracy_cox(hazardsdata, labels):
    # This accuracy is based on estimated survival events against true survival events
    # 中位数大于1的数据标签为1，小于的为0    判断有多少样本的标签和结果标签一样 来计算准确率
    median = np.median(hazardsdata)
    hazards_dichotomize = np.zeros([len(hazardsdata)], dtype=int)
    hazards_dichotomize[hazardsdata > median] = 1
    correct = np.sum(hazards_dichotomize == labels)
    return correct / len(labels)


def auc_cox(hazardsdata, labels):
    fpr, tpr, threshold = roc_curve(labels, hazardsdata)  ###计算真正率和假正率
    roc_auc = auc(fpr, tpr)  ###计算auc的值
    return roc_auc



# 普通的分层采样 bootstrap   计算 C 指数和 ROC 的 auc、sens、spec、acc 的 95% 置信区间
def stratified_pi_bootstrap_metrics(predictions, events_test, time_test, n_iterations=1000):
    """
    使用分层 Bootstrap 计算以下指标的置信区间：
    - C-index
    - AUC
    - Sensitivity
    - Specificity
    - Accuracy
    - Precision
    - F1_score
    
    参数：
    - predictions: 模型预测值（风险得分，越高表示风险越高）
    - events_test: 事件指示数组（1=发生事件，0=删失）
    - time_test: 生存时间数组
    - n_iterations: Bootstrap重采样次数
    
    返回：
    - dict: 包含五个指标的BCa置信区间
    """
    # 分层抽样：按事件/非事件分组
    event_idx = np.where(events_test == 1)[0]
    no_event_idx = np.where(events_test == 0)[0]
    n_event = len(event_idx)
    n_total = len(predictions)


    # 初始化存储结果
    metrics = {
        'Cindex': [],
        'AUC': [],
        'Sensitivity': [],
        'Specificity': [],
        'Accuracy': [],
        'Precision': [],
        'F1_score': []
    }
    

    for _ in range(n_iterations):
        # 分层重采样（保持原始比例）
        sampled_event = resample(event_idx, replace=True, n_samples=n_event)
        sampled_no_event = resample(no_event_idx, replace=True, n_samples=n_total - n_event)
        combined_idx = np.concatenate([sampled_event, sampled_no_event])
        
        # 计算当前样本的所有指标
        c = concordance_index(
            time_test[combined_idx], 
            -predictions[combined_idx],  # 注意方向
            events_test[combined_idx]
        )
        roc_metrics = calculate_2year_metrics(
            predictions[combined_idx],
            events_test[combined_idx],
            time_test[combined_idx]
        )

        # 存储结果
        metrics['Cindex'].append(c)
        metrics['AUC'].append(roc_metrics['AUC'])
        metrics['Sensitivity'].append(roc_metrics['Sensitivity'])
        metrics['Specificity'].append(roc_metrics['Specificity'])
        metrics['Accuracy'].append(roc_metrics['Accuracy'])
        metrics['Precision'].append(roc_metrics['Precision'])
        metrics['F1_score'].append(roc_metrics['F1_score'])


    # # 计算各指标的BCa置信区间
    # return {
    #     k: ci(v, method='pi') for k, v in metrics.items()
    # }
    
    # 计算各指标的置信区间和均值 
    return (
        {k: ci(v, method='pi') for k, v in metrics.items()},  # 置信区间
        {k: np.mean(v) for k, v in metrics.items()}           # 均值
    )


# bca 的分层采样 BCA  bootstrap   计算 C 指数和 ROC 的 auc、sens、spec、acc 的 95% 置信区间
def stratified_bca_bootstrap_metrics(predictions, events_test, time_test, n_iterations=1000):
    """
    使用分层BCa Bootstrap计算以下指标的置信区间：
    - C-index
    - AUC
    - Sensitivity
    - Specificity
    - Accuracy
    - Precision
    - F1_score
    
    参数：
    - predictions: 模型预测值（风险得分，越高表示风险越高）
    - events_test: 事件指示数组（1=发生事件，0=删失）
    - time_test: 生存时间数组
    - n_iterations: Bootstrap重采样次数
    
    返回：
    - dict: 包含五个指标的BCa置信区间
    """
    # 分层抽样：按事件/非事件分组
    event_idx = np.where(events_test == 1)[0]
    no_event_idx = np.where(events_test == 0)[0]
    n_event = len(event_idx)
    n_total = len(predictions)
    
    # 初始化存储结果
    metrics = {
        'Cindex': [],
        'AUC': [],
        'Sensitivity': [],
        'Specificity': [],
        'Accuracy': [],
        'Precision': [],
        'F1_score': []
    }

    for _ in range(n_iterations):
        # 分层重采样（保持原始比例）
        sampled_event = resample(event_idx, replace=True, n_samples=n_event)
        sampled_no_event = resample(no_event_idx, replace=True, n_samples=n_total - n_event)
        combined_idx = np.concatenate([sampled_event, sampled_no_event])
        
        # 计算当前样本的所有指标
        c = concordance_index(
            time_test[combined_idx], 
            -predictions[combined_idx],  # 注意方向
            events_test[combined_idx]
        )
        roc_metrics = calculate_2year_metrics(
            predictions[combined_idx],
            events_test[combined_idx],
            time_test[combined_idx]
        )
        
        # 存储结果
        metrics['Cindex'].append(c)
        metrics['AUC'].append(roc_metrics['AUC'])
        metrics['Sensitivity'].append(roc_metrics['Sensitivity'])
        metrics['Specificity'].append(roc_metrics['Specificity'])
        metrics['Accuracy'].append(roc_metrics['Accuracy'])
        metrics['Precision'].append(roc_metrics['Precision'])
        metrics['F1_score'].append(roc_metrics['F1_score'])

    # # 计算各指标的BCa置信区间
    # return {
    #     k: ci(v, method='bca') for k, v in metrics.items()
    # }
  
    # 计算各指标的置信区间和均值 
    return (
        {k: ci(v, method='bca') for k, v in metrics.items()},  # 置信区间
        {k: np.mean(v) for k, v in metrics.items()}           # 均值
    )


# 计算时间依赖性的 ROC 的相关指标，auc，sens、spec、acc
def calculate_2year_metrics(hazardsdata, labels, survtime):
    """
    计算 x 年无进展生存期（PFS）的 AUC，不修改原始数据。
    
    参数:
        hazardsdata (pd.Series): Cox 模型预测的风险评分。
        labels (pd.Series): 状态（1: 事件发生，0: 删失）。
        survtime (pd.Series): 时间（天）。
        
    返回:
        float: 2 年 xxx 的 AUC 值。
    """
    # 1. 复制数据以避免修改原始数据
    coxpred = hazardsdata.copy()
    labels = labels.copy()
    survtime = survtime.copy()
    
    # 2. 替换无穷值  无穷大变为 最大值+1，无穷小变为 最小值-1
    # 计算最大值和最小值，忽略 NaN 和无穷值
    finite_mask = np.isfinite(coxpred)
    if np.any(finite_mask):
        max_val = np.max(coxpred[finite_mask])
        min_val = np.min(coxpred[finite_mask])
    else:
        max_val = 0
        min_val = 0

    coxpred = np.where(np.isneginf(coxpred), min_val - 1, coxpred)      # 替换负无穷
    coxpred = np.where(np.isinf(coxpred), max_val + 1, coxpred)         # 替换正无穷
    coxpred = np.where(np.isnan(coxpred), 0, coxpred)                   # 替换 NaN

    # 3. 筛选数据：删除随访时间 < 2 年且 xxx_status=0 的样本
    mask = ~((survtime < 730) & (labels == 0))
    filtered_coxpred = coxpred[mask]
    filtered_status = labels[mask]
    filtered_time = survtime[mask]
    
    # 4. 生成 2 年 xxx 标签（1: 2 年内事件发生，0: 2 年内未发生事件）
    two_year_label = (filtered_time < 730) & (filtered_status == 1)
    two_year_label = two_year_label.astype(int)
    
    # 5. 计算 AUC
    auc = roc_auc_score(two_year_label, filtered_coxpred)
    
    # 6. 计算 Sensitivity 和 Specificity
    fpr, tpr, thresholds = roc_curve(two_year_label, filtered_coxpred)
    
    # 使用 Youden's J 统计量选择最佳阈值
    youden_j = tpr - fpr
    best_idx = np.argmax(youden_j)
    best_threshold = thresholds[best_idx]
    
    # 根据最佳阈值计算预测类别
    y_pred = (filtered_coxpred >= best_threshold).astype(int)
    
    # 计算混淆矩阵
    tp = np.sum((y_pred == 1) & (two_year_label == 1))
    fn = np.sum((y_pred == 0) & (two_year_label == 1))
    tn = np.sum((y_pred == 0) & (two_year_label == 0))
    fp = np.sum((y_pred == 1) & (two_year_label == 0))
    
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    specificity = tn / (tn + fp) if (tn + fp) > 0 else np.nan
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else np.nan
    
    # 新增 精确率 和 F1 计算
    precision = tp / (tp + fp) if (tp + fp) > 0 else np.nan
    f1_score = 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else np.nan
    
    return {
        'AUC': auc,                     # ROC曲线下的面积，反映模型在不同阈值下区分正负类的能力
        'Sensitivity': sensitivity,     # 敏感性（召回率）：实际为正的样本中，被正确预测为正的比例。衡量模型捕捉正类的能力（避免漏诊）
        'Specificity': specificity,     # 特异性：实际为负的样本中，被正确预测为负的比例。衡量模型排除负类的能力（避免误诊）
        'Accuracy': accuracy,           # 准确率：所有样本中预测正确的比例（综合衡量整体正确性）
        'Precision': precision,         # 精确率：预测为正的样本中，实际为正的比例（关注假阳性 FP的代价）
        'F1_score': f1_score            # F1分数：精确率和召回率的调和平均数，平衡两者矛盾
    }


def cox_log_rank(hazardsdata, labels, survtime_all):
    median = np.median(hazardsdata)
    hazards_dichotomize = np.zeros([len(hazardsdata)], dtype=int)
    hazards_dichotomize[hazardsdata > median] = 1
    idx = hazards_dichotomize == 0
    T1 = survtime_all[idx]
    T2 = survtime_all[~idx]
    E1 = labels[idx]
    E2 = labels[~idx]
    results = logrank_test(T1, T2, event_observed_A=E1, event_observed_B=E2)
    pvalue_pred = results.p_value
    return(pvalue_pred)


def CoxLoss(survtime, censor, hazard_pred, device):
    # This calculation credit to Travers Ching https://github.com/traversc/cox-nnet
    # Cox-nnet: An artificial neural network method for prognosis prediction of high-throughput omics data
    # current_batch_len = len(survtime)
    # R_mat = np.zeros([current_batch_len, current_batch_len], dtype=int)
    # for i in range(current_batch_len):
    #     for j in range(current_batch_len):
    #         R_mat[i,j] = survtime[j] >= survtime[i]
    i_ = survtime[:,None]       # 将 survtime 扩展为二维数组，i_ 是 survtime 的列向量，j_ 是 survtime 的行向量
    j_ = survtime[None,:]       
    R_mat_ = i_ <= j_           # 通过这两者的比较，可以生成一个布尔矩阵，表示每对样本之间的生存时间关系
    R_mat_ = R_mat_.cpu().numpy()

    R_mat = torch.FloatTensor(R_mat_).to(device)
    theta = hazard_pred.reshape(-1)     # 将 hazard_pred 重塑为一维数组，并将其赋值给 theta
    # theta = theta * 0.01   #This step will make inf number if theta>100?
    exp_theta = torch.exp(theta)    #This step will make inf number if theta>100?   # 计算 theta 的指数值，得到 exp_theta
    loss_cox = -torch.mean((theta - torch.log(torch.sum(exp_theta*R_mat, dim=1))) * censor)    # dim=1,横向压缩，保留列 c = torch.sum(a,dim=1)
    return loss_cox


# lasso 和 lasso2的区别，lasso y0 = cli_train['PFS_status']，lasso2  y0 = cli_train['OS_5s']
def lasso(X0,X1,cli_train,cli_test,epoch,k):
    X_train = X0
    X_test = X1

    y0 = cli_train['PFS_status']

    alphas = np.logspace(-3,1,50)       # 生成 50 个 alpha 值（正则化强度）的数组，这些值之间以对数间隔 10^{-3} 和 10^{1}
 
    # model_lasso = ignore_warnings(LassoCV(alphas = alphas,cv=10,max_iter = 10000).fit)(X_train, y0)
    model_lasso = LassoCV(alphas=alphas, cv=10, max_iter=10000).fit(X_train, y0)        # 通过交叉验证执行 Lasso 回归以找到最佳 alpha, max_iter=10000参数确保了模型有足够的迭代次数来收敛，它适合模型X_train和y0
    # print(model_lasso.alpha_)
    coef = pd.Series(model_lasso.coef_, index = X_train.columns)  # 输出看模型最终选择了几个特征向量，剔除了几个特征向量
    print("Lasso picked " + str(sum(coef != 0)) + " variables and eliminated the other " +  str(sum(coef == 0)) + " variables") # 打印 Lasso 模型选择和消除的特征数量
    index = coef[coef != 0].index
    X_lasso0=X_train[index]
    X_lasso1 = X_test[index]
    X_lasso0=pd.DataFrame(X_lasso0)
    X_lasso1 = pd.DataFrame(X_lasso1)

    X_train_l = pd.concat([X_lasso0,cli_train['PFS_time'],cli_train['PFS_status']],axis = 1)
    X_test_l = pd.concat([X_lasso1, cli_test['PFS_time'], cli_test['PFS_status']], axis=1)

    cox_lasso = CoxPHFitter(penalizer=0.1)
    cox_lasso.fit(X_train_l, 'PFS_time', event_col='PFS_status', step_size=0.5)

    cindex_train = cox_lasso.concordance_index_
    cindex_test = concordance_index(X_test_l['PFS_time'], -cox_lasso.predict_partial_hazard(X_test_l), X_test_l['PFS_status'])

    return cindex_train, cindex_test


def lasso2(X0,X1,cli_train,cli_test,epoch,k):
    X_train = X0
    X_test = X1

    y0 = cli_train['OS_5s']

    alphas = np.logspace(-3,1,50)
    # model_lasso = ignore_warnings(LassoCV(alphas = alphas,cv=10,max_iter = 10000).fit)(X_train, y0)
    model_lasso = LassoCV(alphas=alphas, cv=10, max_iter=10000).fit(X_train, y0)
    print(model_lasso.alpha_)
    coef = pd.Series(model_lasso.coef_, index = X_train.columns)  #输出看模型最终选择了几个特征向量，剔除了几个特征向量
    print("Lasso picked " + str(sum(coef != 0)) + " variables and eliminated the other " +  str(sum(coef == 0)) + " variables")
    index = coef[coef != 0].index
    X_lasso0=X_train[index]
    X_lasso1 = X_test[index]
    X_lasso0=pd.DataFrame(X_lasso0)
    X_lasso1 = pd.DataFrame(X_lasso1)

    X_train_l = pd.concat([X_lasso0,cli_train['PFS_time'],cli_train['PFS_status']],axis = 1)
    X_test_l = pd.concat([X_lasso1, cli_test['PFS_time'], cli_test['PFS_status']], axis=1)

    cox_lasso = CoxPHFitter(penalizer=0.1)
    cox_lasso.fit(X_train_l, 'PFS_time', event_col='PFS_status', step_size=0.5)

    cindex_train = cox_lasso.concordance_index_
    cindex_test = concordance_index(X_test_l['PFS_time'], -cox_lasso.predict_partial_hazard(X_test_l), X_test_l['PFS_status'])

    return cindex_train, cindex_test


def cox_model(X, X2, cli_train, cli_test,pic_path,method,repeat):
    a = pd.concat([X, cli_train['PFS_time'], cli_train['PFS_status']], axis=1)
    aa = pd.concat([X2, cli_test['PFS_time'], cli_test['PFS_status']], axis=1)

    cph = CoxPHFitter(penalizer=0.1)
    cph.fit(a, 'PFS_time', event_col='PFS_status')
    # cph.print_summary()

    c_index_test = concordance_index(aa['PFS_time'], -cph.predict_partial_hazard(aa), aa['PFS_status'])
    pvalue_train = cox_log_rank(cph.predict_partial_hazard(a), a['PFS_status'], a['PFS_time'])
    pvalue_test = cox_log_rank(cph.predict_partial_hazard(aa), aa['PFS_status'], aa['PFS_time'])
    # cph.c_index_test = c_index_test
    draw_km(cph.predict_partial_hazard(aa), cli_test['PFS_status'], cli_test['PFS_time'],
            os.path.join(pic_path, 'test_' + method + str(repeat)))
    draw_km(cph.predict_partial_hazard(a), cli_train['PFS_status'], cli_train['PFS_time'],
            os.path.join(pic_path, 'train_' + method + str(repeat)))

    return cph.predict_partial_hazard(a), cph.predict_partial_hazard(aa), cph.concordance_index_, c_index_test, pvalue_train, pvalue_test


def cox_model_cli(X, X2, cli_train, cli_test,pic_path,method,repeat):

    c = pd.concat([X, cli_train.iloc[:,[1,3,4,6]], cli_train['PFS_time'], cli_train['PFS_status']], axis=1)
    cc = pd.concat([X2, cli_test.iloc[:,[1,3,4,6]], cli_test['PFS_time'], cli_test['PFS_status']], axis=1)

    cph3 = CoxPHFitter(penalizer=0.1)
    cph3.fit(c, 'PFS_time', event_col='PFS_status')
    # cph2.print_summary()
    c_index_test3 = concordance_index(cc['PFS_time'], -cph3.predict_partial_hazard(cc), cc['PFS_status'])
    pvalue_train3 = cox_log_rank(cph3.predict_partial_hazard(c), c['PFS_status'], c['PFS_time'])
    pvalue_test3 = cox_log_rank(cph3.predict_partial_hazard(cc), cc['PFS_status'], cc['PFS_time'])


    draw_km(cph3.predict_partial_hazard(cc), cli_test['PFS_status'], cli_test['PFS_time'],
            os.path.join(pic_path, 'test_' + method + '_cli' + str(repeat)))
    draw_km(cph3.predict_partial_hazard(c), cli_train['PFS_status'], cli_train['PFS_time'],
            os.path.join(pic_path, 'train_' + method + '_cli' + str(repeat)))

    return cph3.predict_partial_hazard(cc), cph3.concordance_index_, c_index_test3, pvalue_train3, pvalue_test3


def cox_model_cli_rad(X, X2, cli_train, cli_test, pic_path, model,repeat):

    a = pd.concat([X, cli_train['PFS_time'], cli_train['PFS_status']], axis=1)
    aa = pd.concat([X2, cli_test['PFS_time'], cli_test['PFS_status']], axis=1)

    cph = CoxPHFitter(penalizer=0.1)
    cph.fit(a, 'PFS_time', event_col='PFS_status')
    # cph.print_summary()
    c_index_test = concordance_index(aa['PFS_time'], -cph.predict_partial_hazard(aa), aa['PFS_status'])
    pvalue_train = cox_log_rank(cph.predict_partial_hazard(a), a['PFS_status'], a['PFS_time'])
    pvalue_test = cox_log_rank(cph.predict_partial_hazard(aa), aa['PFS_status'], aa['PFS_time'])

    c = pd.concat([X, cli_train.iloc[:,[1,3,4,6]], cli_train['PFS_time'], cli_train['PFS_status']], axis=1)
    cc = pd.concat([X2, cli_test.iloc[:,[1,3,4,6]], cli_test['PFS_time'], cli_test['PFS_status']], axis=1)

    cph3 = CoxPHFitter(penalizer=0.1)
    cph3.fit(c, 'PFS_time', event_col='PFS_status')
    # cph2.print_summary()
    c_index_test3 = concordance_index(cc['PFS_time'], -cph3.predict_partial_hazard(cc), cc['PFS_status'])
    pvalue_train3 = cox_log_rank(cph3.predict_partial_hazard(c), c['PFS_status'], c['PFS_time'])
    pvalue_test3 = cox_log_rank(cph3.predict_partial_hazard(cc), cc['PFS_status'], cc['PFS_time'])

    draw_km_test(cph.predict_partial_hazard(a), cph.predict_partial_hazard(aa), cli_test['PFS_status'], cli_test['PFS_time'],
            os.path.join(pic_path, 'test_' + model + str(repeat)))
    draw_km(cph.predict_partial_hazard(a), cli_train['PFS_status'], cli_train['PFS_time'],
            os.path.join(pic_path, 'train_' + model + str(repeat)))

    draw_km_test(cph3.predict_partial_hazard(c), cph3.predict_partial_hazard(cc), cli_test['PFS_status'], cli_test['PFS_time'],
            os.path.join(pic_path, 'test_' + model + '_cli' + str(repeat)))
    draw_km(cph3.predict_partial_hazard(c), cli_train['PFS_status'], cli_train['PFS_time'],
            os.path.join(pic_path, 'train_' + model + '_cli' + str(repeat)))

    return cph.predict_partial_hazard(a), cph.concordance_index_, c_index_test, pvalue_train, pvalue_test, cph.predict_partial_hazard(aa), \
            cph3.predict_partial_hazard(c), cph3.concordance_index_, c_index_test3, pvalue_train3, pvalue_test3, cph3.predict_partial_hazard(cc)


def draw_km(hazardsdata, labels, survtime_all,save_pic_path):
    median = np.median(hazardsdata)
    idx = np.array((hazardsdata.iloc[:] < median), dtype=bool).reshape(-1)      # 根据中位数将hazardsdata分为两部分，其中小于中位数的为低风险组，大于等于中位数的为高风险组
    survtime_all = survtime_all/30.4        # 将生存时间转换为月数

    kmf1 = KaplanMeierFitter()
    kmf1.fit(survtime_all[idx], labels[idx], label='low-risk')  # 用低风险组的数据拟合Kaplan-Meier模型
    ax = kmf1.plot_survival_function(lw = 2, color='green',show_censors=True,figsize=(6.5,5.6))

    kmf2 = KaplanMeierFitter()
    kmf2.fit(survtime_all[~idx], labels[~idx], label='high-risk')
    kmf2.plot_survival_function(lw = 2, color='red',ax=ax,show_censors=True)

    T_1, E_1 = survtime_all[idx], labels[idx]
    T_3, E_3 = survtime_all[~idx], labels[~idx]
    result_logrank = logrank_test(T_1, T_3, event_observed_A=E_1, event_observed_B=E_3)     # 进行Log-rank检验，比较两组的生存曲线差异

    xmajorLocator = MultipleLocator(12)     # 创建一个主刻度定位器，用于设置x轴的刻度间隔
    ax.xaxis.set_major_locator(xmajorLocator)       # 设置x轴的主刻度间隔

    plt.xlabel('Time (months)', fontsize=16)
    plt.ylabel('PFS', fontsize=16)
    plt.legend(loc='lower left', fontsize=12)
    plt.ylim([0.0, 1.05])
    plt.xlim([0, 132])
    if result_logrank.p_value < 0.0001:
        plt.text(2, 0.2, r'p value < 0.0001', fontsize=12)
    else:
        plt.text(2, 0.2, r'p value = %.4f' % result_logrank.p_value, fontsize=12)
    plt.tick_params(labelsize=13)       # 设置刻度标签的字体大小
    add_at_risk_counts(kmf1, kmf2, ax=ax, rows_to_show=['At risk'])  # , rows_to_show=['At risk', 'Censored']
    plt.tight_layout()
    plt.savefig(save_pic_path)

    plt.close()


# # 画 km 曲线时 多保存一个 tif 格式的图片  并在图上显示该模型对应的 C 指数
# def draw_km(hazardsdata, labels, survtime_all, cindex, save_pic_path1, save_pic_path2):
#     median = np.median(hazardsdata)
#     idx = np.array((hazardsdata.iloc[:] < median), dtype=bool).reshape(-1)      # 根据中位数将hazardsdata分为两部分，其中小于中位数的为低风险组，大于等于中位数的为高风险组
#     survtime_all = survtime_all/30.4        # 将生存时间转换为月数

#     kmf1 = KaplanMeierFitter()
#     kmf1.fit(survtime_all[idx], labels[idx], label='Low-risk')  # 用低风险组的数据拟合Kaplan-Meier模型
#     ax = kmf1.plot_survival_function(lw = 2, color='green',show_censors=True,figsize=(6.5,5.6))

#     kmf2 = KaplanMeierFitter()
#     kmf2.fit(survtime_all[~idx], labels[~idx], label='High-risk')
#     kmf2.plot_survival_function(lw = 2, color='red',ax=ax,show_censors=True)

#     T_1, E_1 = survtime_all[idx], labels[idx]
#     T_3, E_3 = survtime_all[~idx], labels[~idx]
#     result_logrank = logrank_test(T_1, T_3, event_observed_A=E_1, event_observed_B=E_3)     # 进行Log-rank检验，比较两组的生存曲线差异

#     xmajorLocator = MultipleLocator(12)     # 创建一个主刻度定位器，用于设置x轴的刻度间隔
#     ax.xaxis.set_major_locator(xmajorLocator)       # 设置x轴的主刻度间隔

#     plt.xlabel('Time (months)', fontsize=16)
#     plt.ylabel('PFS Survival Probability', fontsize=16)
#     plt.legend(loc='lower right', fontsize=12)
#     plt.ylim([0.0, 1.05])
#     plt.xlim([0, 132])
#     # if result_logrank.p_value < 0.0001:
#     #     plt.text(2, 0.2, r'p value < 0.0001', fontsize=12)
#     # else:
#     #     plt.text(2, 0.2, r'p value = %.4f' % result_logrank.p_value, fontsize=12)


#     # 在左下角添加两行文本（p值和C指数）
#     if result_logrank.p_value < 0.0001:
#         p_text = r'Log-rank p value < 0.0001'
#     else:
#         p_text = r'Log-rank p value = %.4f' % result_logrank.p_value

#     # 添加两行文本（调整y坐标位置，避免重叠）
#     plt.text(2, 0.1, p_text, fontsize=12)                # 第一行：p值（原位置）
#     plt.text(2, 0.2, f'C-index = {cindex}', fontsize=12)  # 第二行：C指数（假设c_index已计算）


#     plt.tick_params(labelsize=13)       # 设置刻度标签的字体大小
#     add_at_risk_counts(kmf1, kmf2, ax=ax, rows_to_show=['Number at risk'])  # , rows_to_show=['At risk', 'Censored']
#     plt.tight_layout()
#     plt.savefig(save_pic_path1)
#     plt.savefig(save_pic_path2, dpi=300)

#     plt.close()



def draw_km_test(hazardsdata_train, hazardsdata, labels, survtime_all,save_pic_path):
    median = np.median(hazardsdata_train)
    idx = np.array((hazardsdata.iloc[:] < median), dtype=bool).reshape(-1)
    survtime_all = survtime_all/30.4

    kmf1 = KaplanMeierFitter()
    kmf1.fit(survtime_all[idx], labels[idx], label='low-risk')
    ax = kmf1.plot_survival_function(lw = 2, color='green',show_censors=True,figsize=(6.5,5.6))

    kmf2 = KaplanMeierFitter()
    kmf2.fit(survtime_all[~idx], labels[~idx], label='high-risk')
    kmf2.plot_survival_function(lw = 2, color='red',ax=ax,show_censors=True)

    T_1, E_1 = survtime_all[idx], labels[idx]
    T_3, E_3 = survtime_all[~idx], labels[~idx]
    result_logrank = logrank_test(T_1, T_3, event_observed_A=E_1, event_observed_B=E_3)

    xmajorLocator = MultipleLocator(12)
    ax.xaxis.set_major_locator(xmajorLocator)

    plt.xlabel('Time (months)', fontsize=16)
    plt.ylabel('PFS', fontsize=16)
    plt.legend(loc='lower left', fontsize=12)
    plt.ylim([0.0, 1.05])
    plt.xlim([0, 132])
    if result_logrank.p_value < 0.0001:
        plt.text(2, 0.2, r'p value < 0.0001', fontsize=12)
    else:
        plt.text(2, 0.2, r'p value = %.4f' % result_logrank.p_value, fontsize=12)
    plt.tick_params(labelsize=13)
    add_at_risk_counts(kmf1, kmf2, ax=ax, rows_to_show=['At risk'])  # , rows_to_show=['At risk', 'Censored'] # 添加风险计数，显示在图中
    plt.tight_layout()
    plt.savefig(save_pic_path)

    plt.close()

def get_iAUC(hazardsdata, labels, survtime_all):
    median = np.median(hazardsdata)
    hazardsdata = hazardsdata.values.reshape(-1)
    label = labels.copy()
    time = []
    scores = []

    time_total = list(survtime_all.values)      # 将survtime_all转换为列表
    time_total.sort()               # 对时间列表进行排序
    for time_interval in time_total:        #  遍历排序后的时间列表
        target = survtime_all > time_interval       # 根据当前时间间隔，创建一个布尔索引，标记所有生存时间超过这个时间点的个体
        # label[target]= 0,   # label[~target] = 1
        uncensored = target | labels.astype(bool)       # 创建一个布尔索引，表示未截尾的个体（即生存状态已知的个体）

        # Compute ROC curve and ROC area for each class
        fpr, tpr, threshold = roc_curve(label[uncensored].values.astype(int), hazardsdata[uncensored])  ###计算真正率和假正率
        roc_auc = auc(fpr, tpr)  ###计算auc的值

        # if np.all(labels[uncensored]):
        #     scores.append(0)
        # else:
        if not np.all(labels[uncensored]):
            scores.append(roc_auc)
            time.append(time_interval)
    # plt.ylim(0.5,1.1)
    # plt.plot(time, scores)
    return list([scores, time])

def draw_iAUC(list0,list1,list2,list3,tag,save_pic_path):
    plt.figure(1)  # 创建绘图对象
    plt.ylim(0.5, 1.1)
    plt.plot(list0[1], list0[0], label=tag + '_0', linewidth=1)   #在当前绘图对象绘图（X轴，Y轴，蓝色，线宽度）
    plt.plot(list1[1], list1[0], label=tag + '_1', linewidth=1)
    plt.plot(list2[1], list2[0], label=tag + '_2', linewidth=1)
    plt.plot(list3[1], list3[0], label=tag + '_3', linewidth=1)
    plt.legend()
    plt.xlabel("Time(day)") #X轴标签
    plt.ylabel("AUC")  #Y轴标签
    plt.title("method: " + tag) #图标题
    plt.savefig(save_pic_path) #保存图
    # plt.show()  #显示图
    plt.close()  #显示图

def get_icindex(hazardsdata, status, survtime_all):
    median = np.median(hazardsdata)
    hazardsdata = hazardsdata.values.reshape(-1)

    time = []
    scores = []

    # time_total = list(survtime_all.values)
    time_total = list(survtime_all)
    time_total.sort()
    for time_interval in time_total:
        target = survtime_all > time_interval
        uncensored = target | status.astype(bool)

        # if np.all(status[uncensored]):
        #     scores.append(0)
        # else:
        #     # scores.append(roc_auc_score(status[uncensored], hazardsdata[uncensored]))
        #     scores.append(concordance_index(survtime_all[uncensored], -hazardsdata[uncensored],
        #                       status[uncensored]))
        # time.append(time_interval)
        if not np.all(status[uncensored]):
            scores.append(concordance_index(survtime_all[uncensored], -hazardsdata[uncensored],
                                            status[uncensored]))
            time.append(time_interval)

    return list([scores, time])

# def get_auc(hazardsdata, status, time,time_interval):
#
#     hazardsdata = hazardsdata.values.reshape(-1)
#     label = status.copy()
#
#     target = time > time_interval
#     label[target],label[~target] = 0,1
#     uncensored = target | status.astype(bool)
#
#     # Compute ROC curve and ROC area for each class
#     fpr, tpr, threshold = roc_curve(label[uncensored].values.astype(int), hazardsdata[uncensored])  ###计算真正率和假正率
#     roc_auc = auc(fpr, tpr)  ###计算auc的值
#
#     return roc_auc, fpr, tpr

def get_auc(hazardsdata, status, time,time_interval):
    hazardsdata = hazardsdata.values.reshape(-1)
    label = status.copy()
    fpr, tpr, threshold = roc_curve(label.values.astype(int), hazardsdata)  ### 计算真正率和假正率
    roc_auc = auc(fpr, tpr)  ###计算auc的值

    return roc_auc, fpr, tpr


def draw_dca(hazardsdata, status, time, time_interval,save_pic_path):
    # This accuracy is based on estimated survival events against true survival events
    _range = np.max(hazardsdata) - np.min(hazardsdata)
    hazardsdata_ = (hazardsdata - np.min(hazardsdata)) / _range
    median = np.median(hazardsdata_)
    hazards_dichotomize = np.zeros([len(hazardsdata_)], dtype=int)
    hazards_dichotomize[hazardsdata_ > median] = 1
    label = status.copy()
    target = time > time_interval
    label[target],label[~target] = 0,1
    label = label.values.astype(int)
    uncensored = target | status.astype(bool)

    cnf_matrix = confusion_matrix(label[uncensored], hazardsdata[uncensored])

    morbidity = np.sum(label[uncensored] == 1)/hazardsdata_[uncensored].shape[0]
    pt_arr = []
    net_bnf_arr = []
    jiduan = []
    for i in range(0, 100, 1):
        pt = i / 100
        # compiute TP FP
        hazardsdata_clip = np.zeros(hazardsdata_.shape[0])
        for j in range(hazardsdata_.shape[0]):
            if hazardsdata_[j] >= pt:
                hazardsdata_clip[j] = 1
            else:
                hazardsdata_clip[j] = 0
        cnf_matrix = confusion_matrix(label[uncensored], hazardsdata_clip[uncensored])
        # print(cnf_matrix)
        FP = cnf_matrix[0, 1]
        FN = cnf_matrix[1, 0]
        TP = cnf_matrix[1, 1]
        TN = cnf_matrix[0, 0]

        net_bnf = (TP - (FP * pt / (1 - pt)))/(FP + FN + TP + TN)
        # print('pt {}, TP {}, FP {}, net_bf {}'.format(pt, TP, FP, net_bnf))
        pt_arr.append(pt)
        net_bnf_arr.append(net_bnf)
        jiduan.append(morbidity - (1 - morbidity) * pt / (1 - pt))
    plt.plot(pt_arr, net_bnf_arr, color='red', lw=1, linestyle='--', label='test')
    plt.plot(pt_arr, np.zeros(len(pt_arr)), color='k', lw=1, linestyle='--', label='None')
    plt.plot(pt_arr, jiduan, color='b', lw=1, linestyle='dotted', label='ALL')
    plt.xlim([0.0, 1.0])
    plt.ylim([-1, 1])
    plt.xlabel('Risk Threshold')
    plt.ylabel('Net Benefit')
    plt.title('Train model')
    plt.legend(loc="right")
    plt.savefig(save_pic_path)
    plt.show()
    plt.close()  #显示图


def define_reg(model):      # 封装计算给定模型的正则化损失的过程
    loss_reg = regularize_weights(model=model)
    return loss_reg



def regularize_weights(model):
    l1_reg = None       # 初始化一个变量 l1_reg，用于存储 L1 正则化损失，初始值为 None

    for W in model.parameters():        # 循环遍历模型的所有参数（权重）    model.parameters() 返回一个可迭代的参数列表0/
        if l1_reg is None:
            l1_reg = torch.abs(W).sum()     # 在第一次迭代时，l1_reg 仍然是 None，因此计算第一个权重的 L1 范数
        else:
            l1_reg = l1_reg + torch.abs(W).sum() # torch.abs(W).sum() is equivalent to W.norm(1)
    return l1_reg       # 最终返回计算得到的 L1 正则化损失