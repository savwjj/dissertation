# 训练集数据 划分为训练验证的下标信息 存储成字典保存起来

import pickle
from sklearn.model_selection import KFold   # 用于交叉验证中的K折划分数据集
import datetime
import time
import random



# 对训练集数据进行 五折划分成 训练集：测试集 = 4:1
def train_valid_dividepos(fold,seed):
    CHUS_num = list(range(100))
    HGJ_num = list(range(90))
    MAASTRO_num = list(range(74))
    QIN1_num = list(range(134))
    QIN2_num = list(range(114))
    kf = KFold(n_splits=5, shuffle=True,random_state=seed)
         
    tr_CH,te_CH = [],[]# 初始化KFold
    for train_index, test_index in kf.split(CHUS_num):  # 调用split方法切分数据
        tr_CH.append(train_index)
        te_CH.append(test_index)

    tr_HG,te_HG = [],[]# 初始化KFold
    for train_index, test_index in kf.split(HGJ_num):  # 调用split方法切分数据
        tr_HG.append(train_index)
        te_HG.append(test_index)
    tr_MA,te_MA = [],[]# 初始化KFold
    for train_index, test_index in kf.split(MAASTRO_num):  # 调用split方法切分数据
        tr_MA.append(train_index)
        te_MA.append(test_index)
    tr_Q1,te_Q1 = [],[]# 初始化KFold
    for train_index, test_index in kf.split(QIN1_num):  # 调用split方法切分数据
        tr_Q1.append(train_index)
        te_Q1.append(test_index)
    tr_Q2,te_Q2 = [],[]# 初始化KFold
    for train_index, test_index in kf.split(QIN2_num):  # 调用split方法切分数据
        tr_Q2.append(train_index)
        te_Q2.append(test_index)
    
    train_list = list(tr_CH[fold]) + list(tr_HG[fold] + 100) + list(
        tr_MA[fold] + 190) + list(tr_Q1[fold] + 264) + list(tr_Q2[fold] + 398)
    valid_list = list(te_CH[fold]) + list(te_HG[fold] + 100) + list(
        te_MA[fold] + 190) + list(te_Q1[fold] + 264) + list(te_Q2[fold] + 398)

    return train_list, valid_list


# # 对训练集数据进行 五折划分成 训练集：测试集 = 4:1
# def train_valid_dividepos(fold):
#     # 设置随机种子为当前时间
#     random.seed(int(time.time()))
#     seed = int(time.time())  # 使用当前时间作为种子
    
#     CHUS_num = list(range(100))
#     HGJ_num = list(range(90))
#     MAASTRO_num = list(range(74))
#     QIN1_num = list(range(134))
#     QIN2_num = list(range(114))
#     kf = KFold(n_splits=5, shuffle=True,random_state=seed)
         
#     tr_CH,te_CH = [],[]# 初始化KFold
#     for train_index, test_index in kf.split(CHUS_num):  # 调用split方法切分数据
#         tr_CH.append(train_index)
#         te_CH.append(test_index)

#     tr_HG,te_HG = [],[]# 初始化KFold
#     for train_index, test_index in kf.split(HGJ_num):  # 调用split方法切分数据
#         tr_HG.append(train_index)
#         te_HG.append(test_index)
#     tr_MA,te_MA = [],[]# 初始化KFold
#     for train_index, test_index in kf.split(MAASTRO_num):  # 调用split方法切分数据
#         tr_MA.append(train_index)
#         te_MA.append(test_index)
#     tr_Q1,te_Q1 = [],[]# 初始化KFold
#     for train_index, test_index in kf.split(QIN1_num):  # 调用split方法切分数据
#         tr_Q1.append(train_index)
#         te_Q1.append(test_index)
#     tr_Q2,te_Q2 = [],[]# 初始化KFold
#     for train_index, test_index in kf.split(QIN2_num):  # 调用split方法切分数据
#         tr_Q2.append(train_index)
#         te_Q2.append(test_index)

#     valid_list = list(te_CH[fold]) + list(te_HG[fold] + 100) + list(
#         te_MA[fold] + 190) + list(te_Q1[fold] + 264) + list(te_Q2[fold] + 398)
#     train_list = list(tr_CH[fold]) + list(tr_HG[fold] + 100) + list(
#         tr_MA[fold] + 190) + list(tr_Q1[fold] + 264) + list(tr_Q2[fold] + 398)
#     return valid_list, train_list





# 获取当前日期
print(f"程序开始运行！此时时间为：{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# 缓存数据集训练划分结果的字典
results_tv_div = {}

# 预计算结果并存储到字典中
for rr_temp in range(0, 100, 5):  # 20个随机种子
    for now_fold_temp in range(5):  # 5折交叉验证
        print(f"当前正在处理的是 {rr_temp} 次 的第 {now_fold_temp} 折")
        results_tv_div[(rr_temp, now_fold_temp)] = train_valid_dividepos(now_fold_temp, rr_temp)
print("训练集划分训练验证的字典生成已完成！")

# 将字典保存到文件
with open('/home/fugui/FRGCN/results_tv_div.pkl', 'wb') as f:
    pickle.dump(results_tv_div, f)
print("字典已保存到 results_tv_div.pkl 文件中！")


# 获取当前日期
print(f"程序结束运行！此时时间为：{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")




# # 缓存数据集训练划分结果的字典
# results_tv_div = {}

# print(f"程序开始运行！此时时间为：{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# # 如果需要在后续加载字典，可以使用以下代码：
# with open('/home/fugui/FRGCN/results_tv_div.pkl', 'rb') as f:
#     results_tv_div = pickle.load(f)
# print("字典已从文件加载！")



# # 预计算结果并存储到字典中
# for rr_temp in range(0, 100, 5):
#     for now_fold_temp in range(5):
#         print(f"当前正在处理的是 {rr_temp} 次 的第 {now_fold_temp} 折")
#         print(results_tv_div[(rr_temp, now_fold_temp)])
#         print("\n")
        

# print(f"程序结束运行！此时时间为：{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


















# # 移动训练验证坐标字典

# import shutil
# import os



# datapath = '/home/fugui/FRGCN_temp333/'
# datapath_2 = '/home/fugui/FRGCN_temp444/'

# # my_datasetType_list = ["allColumns", "someColumns1", "someColumns2", 
# #                       "pcaColumns_10", "pcaColumns_20", "pcaColumns_30", "pcaColumns_40", "pcaColumns_50", 
# #                         "pcaColumns_60", "pcaColumns_70", "pcaColumns_80", "pcaColumns_90", "pcaColumns_100"]


# my_datasetType_list = ["allColumns", "someColumns1", "someColumns2", 
#                        "pcaColumns_5", "pcaColumns_10", "pcaColumns_15", "pcaColumns_20", "pcaColumns_25", 
#                        "pcaColumns_30", "pcaColumns_35", "pcaColumns_40", "pcaColumns_45", "pcaColumns_50", 
#                        "pcaColumns_55", "pcaColumns_60", "pcaColumns_65", "pcaColumns_70", "pcaColumns_75", 
#                        "pcaColumns_80", "pcaColumns_85", "pcaColumns_90", "pcaColumns_95", "pcaColumns_100", "pcaColumns_105"]


# src_file_path = os.path.join(datapath, 'results_tv_div.pkl')


# for my_datasetType in my_datasetType_list:
#     dst_file_path = os.path.join(datapath, my_datasetType, 'results_tv_div.pkl')
#     dst2_file_path = os.path.join(datapath_2, my_datasetType, 'results_tv_div.pkl')
#     # 复制文件
#     shutil.copy2(src_file_path, dst_file_path)
#     print(f"复制文件 {src_file_path} 到 {dst_file_path}。")
#     shutil.copy2(src_file_path, dst2_file_path)
#     print(f"复制文件 {src_file_path} 到 {dst2_file_path}。")

#     # 删除文件
#     os.remove(dst_file_path)
#     print(f"删除文件: {dst_file_path}")

#     os.remove(dst2_file_path)
#     print(f"删除文件: {dst2_file_path}")

