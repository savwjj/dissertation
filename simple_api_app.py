# -*- coding: utf-8 -*-
"""
头颈癌生存预测系统 - 移动端适配版本
"""

import os
import sys
import torch
import numpy as np
import pandas as pd
import traceback
from pathlib import Path

# 导入fastapi
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi import Request
from pydantic import BaseModel
import uvicorn

# 从 torch_geometric 导入
from torch_geometric.loader import DataLoader

# ========== 设置路径 ==========
datapath = 'E:/bs/my_system/MPFGNN'
sys.path.append(datapath)

print("="*60)
print("头颈癌生存预测系统 - 移动端版本启动中...")
print(f"工作路径: {datapath}")
print("="*60)

# ========== 导入你的自定义模块 ==========
try:
    from HNC_data import HNC_Dataset_addsr_BN
    from model_ord_GCN import FinalModel
    print("✅ 成功导入自定义模块")
except ImportError as e:
    print(f"❌ 导入自定义模块失败: {e}")
    print(f"当前Python路径: {sys.path}")
    sys.exit(1)

# ========== 配置参数 ==========
device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

my_channel = 128
my_modality = "CT"
my_datasetType = "pcaColumns_10"

# 你选择的5个最佳模型
top5_models = [
    'model_15_0.pth',
    'model_5_0.pth', 
    'model_50_3.pth',
    'model_60_0.pth',
    'model_75_3.pth'
]

# 模型路径
model_base_path = os.path.join(datapath, "final_results", "result_GCN_CT_128_BN", 
                               "detailed_result", "pcaColumns_10", "model")

# ========== 创建FastAPI应用 ==========
app = FastAPI(
    title="头颈癌生存预测系统 - 移动版",
    description="输入病人ID，查询5个模型的预测结果",
    version="1.0.0"
)

# 配置CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ========== 定义数据模型 ==========
class PredictRequest(BaseModel):
    patient_id: str

class PredictResponse(BaseModel):
    patient_id: str
    dataset: str
    predictions: dict
    average_score: float
    risk_group: str

# ========== 加载数据集 ==========
print("\n正在加载数据集...")

try:
    train_dataset = HNC_Dataset_addsr_BN(
        my_modality, 
        root=os.path.join(datapath, my_datasetType, "dataset"), 
        datapath=os.path.join(datapath, my_datasetType), 
        data_set=0
    )
    
    test_dataset = HNC_Dataset_addsr_BN(
        my_modality, 
        root=os.path.join(datapath, my_datasetType, "dataset"), 
        datapath=os.path.join(datapath, my_datasetType), 
        data_set=1
    )
    
    testsr_dataset = HNC_Dataset_addsr_BN(
        my_modality, 
        root=os.path.join(datapath, my_datasetType, "dataset"), 
        datapath=os.path.join(datapath, my_datasetType), 
        data_set=2
    )
    
    print(f"✅ 训练集大小: {len(train_dataset)}")
    print(f"✅ 测试集大小: {len(test_dataset)}")
    print(f"✅ 子区域测试集大小: {len(testsr_dataset)}")
    
except Exception as e:
    print(f"❌ 加载数据集失败: {e}")
    traceback.print_exc()
    sys.exit(1)

# ========== 加载模型 ==========
print("\n正在加载5个模型...")
models = {}

for i, model_name in enumerate(top5_models):
    try:
        print(f"加载模型 {i+1}/5: {model_name}")
        
        model_path_full = os.path.join(model_base_path, model_name)
        if not os.path.exists(model_path_full):
            print(f"  ⚠️ 模型文件不存在: {model_path_full}")
            continue
            
        model = FinalModel(my_channel, train_dataset.num_features).to(device)
        model.load_state_dict(torch.load(model_path_full, map_location=device))
        model.eval()
        
        models[model_name] = model
        print(f"  ✅ 加载成功")
        
    except Exception as e:
        print(f"  ❌ 加载失败: {e}")

print(f"\n✅ 成功加载 {len(models)} 个模型！")

# ========== 创建病人ID映射表 ==========
def create_patient_id_map():
    """创建病人ID的查找表"""
    id_map = {}
    
    print("\n正在创建病人ID映射表...")
    
    # 处理训练集
    train_ids = set()
    for i, data in enumerate(train_dataset):
        if hasattr(data, 'patient_id'):
            pid = data.patient_id
            train_ids.add(pid)
            id_map[pid] = {
                'dataset': 'train',
                'index': i
            }
    
    # 处理测试集
    test_ids = set()
    test_row_count = 0
    for i, data in enumerate(test_dataset):
        test_row_count += 1
        if hasattr(data, 'patient_id'):
            pid = data.patient_id
            test_ids.add(pid)
            if pid not in id_map:
                id_map[pid] = {
                    'dataset': 'test',
                    'index': i
                }
    
    # 处理子区域测试集
    testsr_ids = set()
    for i, data in enumerate(testsr_dataset):
        if hasattr(data, 'patient_id'):
            pid = data.patient_id
            testsr_ids.add(pid)
            if pid not in id_map:
                id_map[pid] = {
                    'dataset': 'testsr',
                    'index': i
                }
    
    print(f"\n📊 详细统计信息:")
    print(f"训练集唯一病人数: {len(train_ids)}")
    print(f"测试集总行数: {test_row_count}")
    print(f"测试集唯一病人数: {len(test_ids)}")
    print(f"子区域测试集唯一病人数: {len(testsr_ids)}")
    print(f"总唯一病人数: {len(id_map)}")
    
    return id_map

# 创建映射表
patient_id_map = create_patient_id_map()

# ========== 定义预测函数 ==========
def predict_patient(patient_id):
    """预测单个病人的风险分数"""
    
    # 查找病人
    if patient_id not in patient_id_map:
        return None, f"未找到病人ID: {patient_id}"
    
    patient_info = patient_id_map[patient_id]
    dataset_name = patient_info['dataset']
    data_index = patient_info['index']
    
    print(f"找到病人: {patient_id}, 数据集: {dataset_name}, 索引: {data_index}")
    
    # 根据数据集获取对应的数据
    if dataset_name == 'train':
        data = train_dataset[data_index]
    elif dataset_name == 'test':
        data = test_dataset[data_index]
    else:
        data = testsr_dataset[data_index]
    
    # 创建数据加载器
    loader = DataLoader([data], batch_size=1)
    
    # 用5个模型分别预测
    predictions = {}
    with torch.no_grad():
        for model_name, model in models.items():
            try:
                # 获取数据
                for batch_data in loader:
                    batch_data = batch_data.to(device)
                    
                    # 预测
                    pred = model(batch_data)
                    
                    # 处理NaN
                    pred[pred.isnan()] = 0
                    
                    # 转换为Python数值
                    pred_numpy = pred.detach().cpu().numpy()
                    
                    # 根据形状选择正确的索引方式
                    if len(pred_numpy.shape) == 1:
                        score = float(pred_numpy[0])
                    elif len(pred_numpy.shape) == 2:
                        score = float(pred_numpy[0, 0])
                    else:
                        score = float(pred_numpy.flatten()[0])
                    
                    predictions[model_name] = score
                    break  # 只取第一个batch
                    
            except Exception as e:
                print(f"模型 {model_name} 预测失败: {e}")
                continue
    
    if not predictions:
        return None, "没有模型可用"
    
    # 计算平均分
    avg_score = np.mean(list(predictions.values()))
    
    # 确定风险分组
    risk_group = "高危" if avg_score > 0 else "低危"
    
    print(f"预测成功: 平均分={avg_score:.6f}, 分组={risk_group}")
    
    return {
        'patient_id': patient_id,
        'dataset': dataset_name,
        'predictions': predictions,
        'average_score': float(avg_score),
        'risk_group': risk_group
    }, None

# ========== 全局异常处理 ==========
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """全局异常处理"""
    print(f"全局异常: {exc}")
    traceback.print_exc()
    return JSONResponse(
        status_code=500,
        content={"detail": f"服务器内部错误: {str(exc)}"}
    )

# ========== API路由 ==========
@app.get("/", response_class=HTMLResponse)
async def root():
    """返回移动端优化的网页界面"""
    html_content = """
    <!DOCTYPE html>
    <html lang="zh-CN">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=yes">
        <meta name="apple-mobile-web-app-capable" content="yes">
        <meta name="apple-mobile-web-app-status-bar-style" content="black-translucent">
        <title>头颈癌生存预测·移动版</title>
        <style>
            /* 全局样式 - 移动优先 */
            * {
                margin: 0;
                padding: 0;
                box-sizing: border-box;
                -webkit-tap-highlight-color: transparent;
            }
            
            body {
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
                background: linear-gradient(145deg, #f6f9fc 0%, #e6f0f9 100%);
                min-height: 100vh;
                padding: 12px;
                color: #2c3e50;
            }
            
            .app-container {
                max-width: 500px;
                margin: 0 auto;
                padding-bottom: 20px;
            }
            
            /* 头部卡片 */
            .header-card {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                border-radius: 24px;
                padding: 24px 20px;
                margin-bottom: 16px;
                color: white;
                box-shadow: 0 10px 25px rgba(102, 126, 234, 0.3);
            }
            
            .header-title {
                font-size: 24px;
                font-weight: 600;
                margin-bottom: 4px;
                letter-spacing: 0.5px;
            }
            
            .header-subtitle {
                font-size: 14px;
                opacity: 0.9;
                margin-bottom: 20px;
            }
            
            /* 统计卡片网格 */
            .stats-grid {
                display: grid;
                grid-template-columns: 1fr 1fr 1fr 1fr;
                gap: 8px;
                margin-top: 8px;
            }
            
            .stat-card {
                background: rgba(255, 255, 255, 0.2);
                backdrop-filter: blur(10px);
                border-radius: 16px;
                padding: 12px 4px;
                text-align: center;
                border: 1px solid rgba(255, 255, 255, 0.1);
            }
            
            .stat-value {
                font-size: 20px;
                font-weight: 700;
                margin-bottom: 4px;
            }
            
            .stat-label {
                font-size: 11px;
                opacity: 0.9;
                line-height: 1.3;
            }
            
            /* 搜索卡片 */
            .search-card {
                background: white;
                border-radius: 24px;
                padding: 20px;
                margin-bottom: 16px;
                box-shadow: 0 5px 20px rgba(0, 0, 0, 0.05);
            }
            
            .search-title {
                font-size: 16px;
                font-weight: 600;
                margin-bottom: 16px;
                color: #4a5568;
                display: flex;
                align-items: center;
                gap: 8px;
            }
            
            .search-title i {
                font-size: 20px;
            }
            
            .input-wrapper {
                position: relative;
                margin-bottom: 16px;
            }
            
            .search-input {
                width: 100%;
                padding: 16px 20px;
                font-size: 16px;
                border: 2px solid #e2e8f0;
                border-radius: 20px;
                outline: none;
                transition: all 0.3s;
                background: #f8fafc;
            }
            
            .search-input:focus {
                border-color: #667eea;
                background: white;
                box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
            }
            
            .search-button {
                width: 100%;
                padding: 16px;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                border: none;
                border-radius: 20px;
                font-size: 16px;
                font-weight: 600;
                cursor: pointer;
                transition: transform 0.2s, box-shadow 0.2s;
                box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
                margin-bottom: 16px;
            }
            
            .search-button:active {
                transform: scale(0.98);
                box-shadow: 0 2px 10px rgba(102, 126, 234, 0.2);
            }
            
            .search-button:disabled {
                opacity: 0.6;
                pointer-events: none;
            }
            
            /* 标签云 */
            .tags-title {
                font-size: 14px;
                color: #718096;
                margin-bottom: 12px;
                display: flex;
                align-items: center;
                gap: 6px;
            }
            
            .tags-container {
                display: flex;
                flex-wrap: wrap;
                gap: 8px;
            }
            
            .patient-tag {
                background: #f0f4f8;
                padding: 8px 16px;
                border-radius: 30px;
                font-size: 14px;
                color: #4a5568;
                cursor: pointer;
                transition: all 0.2s;
                border: 1px solid #e2e8f0;
                box-shadow: 0 2px 5px rgba(0,0,0,0.02);
            }
            
            .patient-tag:active {
                background: #667eea;
                color: white;
                border-color: #667eea;
                transform: scale(0.96);
            }
            
            /* 加载动画 */
            .loading-container {
                background: white;
                border-radius: 24px;
                padding: 40px 20px;
                text-align: center;
                margin-bottom: 16px;
                display: none;
            }
            
            .spinner {
                width: 50px;
                height: 50px;
                border: 4px solid #f0f4f8;
                border-top: 4px solid #667eea;
                border-radius: 50%;
                animation: spin 1s linear infinite;
                margin: 0 auto 16px;
            }
            
            @keyframes spin {
                0% { transform: rotate(0deg); }
                100% { transform: rotate(360deg); }
            }
            
            .loading-text {
                color: #718096;
                font-size: 15px;
            }
            
            /* 结果卡片 */
            .result-card {
                background: white;
                border-radius: 24px;
                padding: 20px;
                margin-bottom: 16px;
                box-shadow: 0 5px 20px rgba(0, 0, 0, 0.05);
                display: none;
                animation: slideUp 0.3s ease;
            }
            
            @keyframes slideUp {
                from {
                    opacity: 0;
                    transform: translateY(20px);
                }
                to {
                    opacity: 1;
                    transform: translateY(0);
                }
            }
            
            .result-header {
                display: flex;
                align-items: center;
                gap: 12px;
                margin-bottom: 20px;
                padding-bottom: 16px;
                border-bottom: 2px solid #f0f4f8;
            }
            
            .result-icon {
                width: 48px;
                height: 48px;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                border-radius: 16px;
                display: flex;
                align-items: center;
                justify-content: center;
                color: white;
                font-size: 24px;
            }
            
            .result-title {
                flex: 1;
            }
            
            .result-title h3 {
                font-size: 16px;
                color: #718096;
                margin-bottom: 4px;
            }
            
            .result-title h2 {
                font-size: 20px;
                color: #2d3748;
            }
            
            /* 信息行 */
            .info-row {
                display: flex;
                justify-content: space-between;
                align-items: center;
                padding: 12px 0;
                border-bottom: 1px solid #f0f4f8;
            }
            
            .info-label {
                color: #718096;
                font-size: 14px;
                display: flex;
                align-items: center;
                gap: 6px;
            }
            
            .info-value {
                font-weight: 600;
                font-size: 16px;
                color: #2d3748;
            }
            
            .risk-high {
                color: #e53e3e;
                background: #fff5f5;
                padding: 4px 12px;
                border-radius: 30px;
                font-weight: 600;
            }
            
            .risk-low {
                color: #38a169;
                background: #f0fff4;
                padding: 4px 12px;
                border-radius: 30px;
                font-weight: 600;
            }
            
            /* 模型列表 */
            .models-section {
                margin-top: 20px;
            }
            
            .models-title {
                font-size: 16px;
                font-weight: 600;
                color: #4a5568;
                margin-bottom: 12px;
                display: flex;
                align-items: center;
                gap: 8px;
            }
            
            .model-item {
                display: flex;
                justify-content: space-between;
                align-items: center;
                padding: 12px;
                background: #f8fafc;
                border-radius: 16px;
                margin-bottom: 8px;
                transition: transform 0.2s;
            }
            
            .model-item:active {
                transform: scale(0.99);
                background: #edf2f7;
            }
            
            .model-name {
                font-weight: 500;
                color: #4a5568;
            }
            
            .model-score {
                font-weight: 700;
                color: #667eea;
                font-size: 16px;
            }
            
            /* 错误提示 */
            .error-card {
                background: #fff5f5;
                border: 1px solid #feb2b2;
                border-radius: 24px;
                padding: 20px;
                margin-bottom: 16px;
                color: #c53030;
                display: none;
                animation: slideUp 0.3s ease;
            }
            
            .error-content {
                display: flex;
                align-items: center;
                gap: 12px;
            }
            
            .error-icon {
                font-size: 24px;
            }
            
            .error-message {
                flex: 1;
                font-size: 15px;
                line-height: 1.4;
            }
            
            /* 底部 */
            .footer {
                text-align: center;
                padding: 20px;
                color: #a0aec0;
                font-size: 12px;
            }
            
            /* 工具类 */
            .text-center { text-align: center; }
            .mt-16 { margin-top: 16px; }
            .mb-8 { margin-bottom: 8px; }
        </style>
    </head>
    <body>
        <div class="app-container">
            <!-- 头部卡片 -->
            <div class="header-card">
                <div class="header-title">🔬 头颈癌预测</div>
                <div class="header-subtitle">基于5个GCN模型的集成预测</div>
                
                <div class="stats-grid">
                    <div class="stat-card">
                        <div class="stat-value" id="totalPatients">-</div>
                        <div class="stat-label">总病人数</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-value" id="trainCount">-</div>
                        <div class="stat-label">训练集</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-value" id="testCount">-</div>
                        <div class="stat-label">测试集</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-value">5</div>
                        <div class="stat-label">模型数</div>
                    </div>
                </div>
            </div>
            
            <!-- 搜索卡片 -->
            <div class="search-card">
                <div class="search-title">
                    <span>🔍</span>
                    <span>病人查询</span>
                </div>
                
                <div class="input-wrapper">
                    <input type="text" class="search-input" id="patientId" 
                           placeholder="输入病人ID" value="HN-CHUS-001">
                </div>
                
                <button class="search-button" id="searchBtn" onclick="predict()">
                    开始预测
                </button>
                
                <div class="tags-title">
                    <span>📋</span>
                    <span>示例病人ID</span>
                </div>
                
                <div class="tags-container" id="exampleIds">
                    <!-- 动态加载示例ID -->
                </div>
            </div>
            
            <!-- 加载动画 -->
            <div class="loading-container" id="loading">
                <div class="spinner"></div>
                <div class="loading-text">AI模型推理中，请稍候...</div>
            </div>
            
            <!-- 错误提示 -->
            <div class="error-card" id="error">
                <div class="error-content">
                    <div class="error-icon">⚠️</div>
                    <div class="error-message" id="errorMessage"></div>
                </div>
            </div>
            
            <!-- 结果卡片 -->
            <div class="result-card" id="result">
                <div class="result-header">
                    <div class="result-icon">📊</div>
                    <div class="result-title">
                        <h3>预测结果</h3>
                        <h2 id="resultPatientId">-</h2>
                    </div>
                </div>
                
                <div class="info-row">
                    <span class="info-label">📁 所属数据集</span>
                    <span class="info-value" id="resultDataset"></span>
                </div>
                
                <div class="info-row">
                    <span class="info-label">📈 平均风险评分</span>
                    <span class="info-value" id="resultAvgScore"></span>
                </div>
                
                <div class="info-row">
                    <span class="info-label">⚠️ 风险分组</span>
                    <span class="info-value" id="resultRiskGroup"></span>
                </div>
                
                <div class="models-section">
                    <div class="models-title">
                        <span>🤖</span>
                        <span>各模型预测详情</span>
                    </div>
                    <div id="modelDetails"></div>
                </div>
            </div>
            
            <div class="footer">
                ⚕️ 仅供研究使用 · 5个GCN模型集成
            </div>
        </div>

        <script>
            // 页面加载时初始化
            window.onload = function() {
                loadStats();
                loadExampleIds();
            };
            
            // 加载统计数据
            async function loadStats() {
                try {
                    const response = await fetch('/patients');
                    const data = await response.json();
                    
                    document.getElementById('totalPatients').textContent = data.total;
                    document.getElementById('trainCount').textContent = data.datasets?.train || '-';
                    document.getElementById('testCount').textContent = data.datasets?.test || '-';
                } catch (error) {
                    console.error('加载统计数据失败:', error);
                }
            }
            
            // 加载示例ID
            async function loadExampleIds() {
                try {
                    const response = await fetch('/patients?limit=12');
                    const data = await response.json();
                    
                    const container = document.getElementById('exampleIds');
                    container.innerHTML = '';
                    
                    if (data.patients && data.patients.length > 0) {
                        // 显示前12个示例
                        data.patients.slice(0, 12).forEach(pid => {
                            const tag = document.createElement('span');
                            tag.className = 'patient-tag';
                            tag.textContent = pid;
                            tag.onclick = () => {
                                document.getElementById('patientId').value = pid;
                                predict();
                            };
                            container.appendChild(tag);
                        });
                    } else {
                        // 默认示例
                        ['HN-CHUS-001', 'HN-CHUS-002-ok', 'HN-CHUS-003', 'HN-CHUS-004'].forEach(pid => {
                            const tag = document.createElement('span');
                            tag.className = 'patient-tag';
                            tag.textContent = pid;
                            tag.onclick = () => {
                                document.getElementById('patientId').value = pid;
                                predict();
                            };
                            container.appendChild(tag);
                        });
                    }
                } catch (error) {
                    console.error('加载示例ID失败:', error);
                }
            }
            
            // 预测函数
            async function predict() {
                const patientId = document.getElementById('patientId').value.trim();
                const searchBtn = document.getElementById('searchBtn');
                
                if (!patientId) {
                    showError('请输入病人ID');
                    return;
                }
                
                // 显示加载，隐藏结果和错误
                document.getElementById('loading').style.display = 'block';
                document.getElementById('result').style.display = 'none';
                document.getElementById('error').style.display = 'none';
                searchBtn.disabled = true;
                
                try {
                    const response = await fetch('/predict', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify({patient_id: patientId})
                    });
                    
                    if (!response.ok) {
                        const errorData = await response.json();
                        throw new Error(errorData.detail || `HTTP错误: ${response.status}`);
                    }
                    
                    const data = await response.json();
                    
                    // 显示结果
                    document.getElementById('resultPatientId').textContent = data.patient_id;
                    document.getElementById('resultDataset').textContent = data.dataset;
                    document.getElementById('resultAvgScore').textContent = data.average_score.toFixed(6);
                    
                    const riskSpan = document.getElementById('resultRiskGroup');
                    riskSpan.textContent = data.risk_group;
                    riskSpan.className = data.risk_group === '高危' ? 'risk-high' : 'risk-low';
                    
                    // 显示各模型预测
                    const modelDiv = document.getElementById('modelDetails');
                    modelDiv.innerHTML = '';
                    
                    const sortedModels = Object.entries(data.predictions)
                        .sort((a, b) => b[1] - a[1]);
                    
                    sortedModels.forEach(([model, score]) => {
                        const item = document.createElement('div');
                        item.className = 'model-item';
                        
                        let shortName = model.replace('model_', '').replace('.pth', '');
                        
                        item.innerHTML = `
                            <span class="model-name">模型 ${shortName}</span>
                            <span class="model-score">${score.toFixed(6)}</span>
                        `;
                        modelDiv.appendChild(item);
                    });
                    
                    document.getElementById('result').style.display = 'block';
                    
                } catch (error) {
                    console.error('预测错误:', error);
                    showError(error.message);
                } finally {
                    document.getElementById('loading').style.display = 'none';
                    searchBtn.disabled = false;
                }
            }
            
            // 显示错误
            function showError(message) {
                document.getElementById('errorMessage').textContent = message;
                document.getElementById('error').style.display = 'block';
            }
            
            // 按回车键查询
            document.getElementById('patientId').addEventListener('keypress', function(e) {
                if (e.key === 'Enter') {
                    predict();
                }
            });
            
            // 触摸优化
            document.querySelectorAll('button, .patient-tag').forEach(el => {
                el.addEventListener('touchstart', function() {
                    this.style.opacity = '0.8';
                });
                el.addEventListener('touchend', function() {
                    this.style.opacity = '1';
                });
            });
        </script>
    </body>
    </html>
    """
    return html_content

@app.post("/predict")
async def predict(request: PredictRequest):
    """预测API"""
    try:
        print(f"收到预测请求: {request.patient_id}")
        result, error = predict_patient(request.patient_id)
        
        if error:
            print(f"预测失败: {error}")
            raise HTTPException(status_code=404, detail=error)
        
        print(f"预测成功: {result['average_score']}")
        return result
        
    except Exception as e:
        print(f"预测过程异常: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/patients")
async def list_patients(limit: int = 20):
    """列出病人ID"""
    try:
        all_patients = list(patient_id_map.keys())
        
        # 按数据集统计
        datasets = {'train': 0, 'test': 0, 'testsr': 0}
        for info in patient_id_map.values():
            datasets[info['dataset']] += 1
        
        return {
            "total": len(all_patients),
            "patients": all_patients[:limit],
            "datasets": datasets
        }
    except Exception as e:
        print(f"获取病人列表失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """健康检查"""
    return {
        "status": "healthy",
        "models_loaded": len(models),
        "patients_loaded": len(patient_id_map),
        "device": str(device)
    }

# ========== 启动服务器 ==========
if __name__ == "__main__":
    print("\n" + "="*60)
    print("✅ 移动端版本启动完成！")
    print(f"📊 模型数量: {len(models)}")
    print(f"👥 唯一病人数: {len(patient_id_map)}")
    print(f"💻 设备: {device}")
    print("="*60)
    print("\n🌐 访问地址:")
    print("   📱 手机: http://<你的IP>:8000")
    print("   💻 电脑: http://localhost:8000")
    print("="*60 + "\n")
    
    # 启动服务器
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )