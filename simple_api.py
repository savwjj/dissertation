# -*- coding: utf-8 -*-
"""
头颈癌生存预测系统 - 修复数组索引问题
"""

import os
import sys
import torch
import numpy as np
import pandas as pd
import traceback
from pathlib import Path
import glob

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
print("头颈癌生存预测系统启动中...")
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
    title="头颈癌生存预测系统",
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
    
    # 从你的截图看到，病人ID格式是 HN-CHUS-001 这样的
    # 处理训练集
    print("处理训练集...")
    for i, data in enumerate(train_dataset):
        # 尝试多种方式获取patient_id
        pid = None
        
        # 方式1: 直接属性
        if hasattr(data, 'patient_id'):
            pid = data.patient_id
        # 方式2: 如果数据是字典
        elif isinstance(data, dict) and 'patient_id' in data:
            pid = data['patient_id']
        # 方式3: 如果数据有__dict__
        elif hasattr(data, '__dict__') and 'patient_id' in data.__dict__:
            pid = data.__dict__['patient_id']
        
        if pid is None:
            pid = f"TRAIN_{i:04d}"
            
        id_map[pid] = {
            'dataset': 'train',
            'index': i
        }
    
    # 处理测试集
    print("处理测试集...")
    for i, data in enumerate(test_dataset):
        if hasattr(data, 'patient_id'):
            pid = data.patient_id
        else:
            pid = f"TEST_{i:04d}"
            
        id_map[pid] = {
            'dataset': 'test',
            'index': i
        }
    
    # 处理子区域测试集
    print("处理子区域测试集...")
    for i, data in enumerate(testsr_dataset):
        if hasattr(data, 'patient_id'):
            pid = data.patient_id
        else:
            pid = f"TESTSR_{i:04d}"
            
        id_map[pid] = {
            'dataset': 'testsr',
            'index': i
        }
    
    print(f"✅ 共创建 {len(id_map)} 个病人ID映射")
    
    # 显示所有病人ID
    print("\n所有可查询的病人ID:")
    all_pids = list(id_map.keys())
    for i, pid in enumerate(all_pids[:20]):  # 只显示前20个
        print(f"  {i+1}. {pid} (数据集: {id_map[pid]['dataset']})")
    if len(all_pids) > 20:
        print(f"     ... 还有 {len(all_pids) - 20} 个")
    
    return id_map

# 创建映射表
patient_id_map = create_patient_id_map()

# ========== 定义预测函数 - 修复版本 ==========
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
                    
                    # 转换为Python数值 - 修复索引问题
                    pred_numpy = pred.detach().cpu().numpy()
                    
                    # 打印预测值的形状，用于调试
                    print(f"模型 {model_name} 预测形状: {pred_numpy.shape}")
                    
                    # 根据形状选择正确的索引方式
                    if len(pred_numpy.shape) == 1:
                        # 一维数组，直接取第一个元素
                        score = float(pred_numpy[0])
                    elif len(pred_numpy.shape) == 2:
                        # 二维数组，取第一行第一列
                        score = float(pred_numpy[0, 0])
                    else:
                        # 其他情况，取第一个元素
                        score = float(pred_numpy.flatten()[0])
                    
                    predictions[model_name] = score
                    break  # 只取第一个batch
                    
            except Exception as e:
                print(f"模型 {model_name} 预测失败: {e}")
                traceback.print_exc()
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
    """返回网页界面"""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>头颈癌生存预测系统</title>
        <meta charset="utf-8">
        <style>
            body { 
                font-family: 'Microsoft YaHei', Arial, sans-serif; 
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                margin: 0; 
                padding: 20px;
                min-height: 100vh;
            }
            .container { 
                max-width: 800px; 
                margin: 0 auto; 
                background: white; 
                border-radius: 20px; 
                padding: 30px; 
                box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            }
            h1 { 
                color: #333; 
                text-align: center;
                margin-bottom: 10px;
            }
            .subtitle {
                text-align: center;
                color: #666;
                margin-bottom: 30px;
            }
            .stats {
                display: flex;
                justify-content: space-around;
                margin-bottom: 30px;
                padding: 20px;
                background: #f8f9fa;
                border-radius: 10px;
            }
            .stat-item {
                text-align: center;
            }
            .stat-value {
                font-size: 24px;
                font-weight: bold;
                color: #667eea;
            }
            .stat-label {
                color: #666;
                font-size: 14px;
                margin-top: 5px;
            }
            .search-box {
                background: #f8f9fa;
                border-radius: 10px;
                padding: 20px;
                margin-bottom: 20px;
            }
            .input-group {
                display: flex;
                gap: 10px;
                margin-top: 10px;
            }
            input { 
                flex: 1;
                padding: 12px; 
                border: 2px solid #e0e0e0; 
                border-radius: 8px; 
                font-size: 16px; 
                transition: border-color 0.3s;
            }
            input:focus {
                outline: none;
                border-color: #667eea;
            }
            button { 
                padding: 12px 30px; 
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white; 
                border: none; 
                border-radius: 8px; 
                cursor: pointer; 
                font-size: 16px;
                transition: transform 0.2s;
            }
            button:hover { 
                transform: translateY(-2px);
                box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
            }
            .example-list {
                display: flex;
                flex-wrap: wrap;
                gap: 10px;
                margin-top: 15px;
            }
            .example-tag {
                padding: 5px 15px;
                background: #e0e0e0;
                border-radius: 20px;
                cursor: pointer;
                font-size: 14px;
                transition: background 0.3s;
            }
            .example-tag:hover {
                background: #667eea;
                color: white;
            }
            .result { 
                margin-top: 20px; 
                padding: 20px; 
                background: #f8f9fa; 
                border-radius: 10px; 
                display: none; 
            }
            .result h2 {
                color: #333;
                margin-top: 0;
                border-bottom: 2px solid #e0e0e0;
                padding-bottom: 10px;
            }
            .info-row {
                display: flex;
                justify-content: space-between;
                padding: 10px;
                background: white;
                margin: 5px 0;
                border-radius: 5px;
            }
            .info-label {
                color: #666;
                font-weight: 500;
            }
            .info-value {
                font-weight: bold;
                color: #333;
            }
            .high-risk { color: #dc3545; font-weight: bold; }
            .low-risk { color: #28a745; font-weight: bold; }
            .model-list {
                margin-top: 20px;
            }
            .model-item {
                display: flex;
                justify-content: space-between;
                padding: 8px;
                border-bottom: 1px solid #e0e0e0;
            }
            .error { 
                color: #dc3545; 
                padding: 10px; 
                background: #f8d7da; 
                border-radius: 5px; 
                margin-top: 10px; 
                display: none; 
            }
            .loading { 
                display: none; 
                text-align: center; 
                padding: 20px;
            }
            .spinner {
                border: 4px solid #f3f3f3;
                border-top: 4px solid #667eea;
                border-radius: 50%;
                width: 40px;
                height: 40px;
                animation: spin 1s linear infinite;
                margin: 0 auto 10px;
            }
            @keyframes spin {
                0% { transform: rotate(0deg); }
                100% { transform: rotate(360deg); }
            }
            .footer {
                text-align: center;
                padding: 20px;
                color: #666;
                font-size: 14px;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🔬 头颈癌生存预测系统</h1>
            <div class="subtitle">基于5个GCN模型的集成预测</div>
            
            <div class="stats">
                <div class="stat-item">
                    <div class="stat-value" id="totalPatients">-</div>
                    <div class="stat-label">总病人数</div>
                </div>
                <div class="stat-item">
                    <div class="stat-value" id="trainCount">-</div>
                    <div class="stat-label">训练集</div>
                </div>
                <div class="stat-item">
                    <div class="stat-value" id="testCount">-</div>
                    <div class="stat-label">测试集</div>
                </div>
                <div class="stat-item">
                    <div class="stat-value">5</div>
                    <div class="stat-label">模型数</div>
                </div>
            </div>
            
            <div class="search-box">
                <h3>🔍 病人查询</h3>
                <div class="input-group">
                    <input type="text" id="patientId" placeholder="请输入病人ID (例如: HN-CHUS-001)" value="HN-CHUS-001">
                    <button onclick="predict()">查询</button>
                </div>
                
                <div class="example-list" id="exampleIds">
                    <!-- 示例ID会动态加载 -->
                </div>
            </div>
            
            <div id="loading" class="loading">
                <div class="spinner"></div>
                <p>⏳ 正在预测中，请稍候...</p>
            </div>
            
            <div id="error" class="error"></div>
            
            <div id="result" class="result">
                <h2>📊 预测结果</h2>
                <div class="info-row">
                    <span class="info-label">病人ID：</span>
                    <span class="info-value" id="resultPatientId"></span>
                </div>
                <div class="info-row">
                    <span class="info-label">所属数据集：</span>
                    <span class="info-value" id="resultDataset"></span>
                </div>
                <div class="info-row">
                    <span class="info-label">平均风险评分：</span>
                    <span class="info-value" id="resultAvgScore"></span>
                </div>
                <div class="info-row">
                    <span class="info-label">风险分组：</span>
                    <span class="info-value" id="resultRiskGroup"></span>
                </div>
                
                <h3 style="margin-top: 20px;">各模型预测详情：</h3>
                <div class="model-list" id="modelDetails"></div>
            </div>
            
            <div class="footer">
                ⚕️ 本系统仅供研究使用 | 基于5个GCN模型的集成预测
            </div>
        </div>

        <script>
            // 页面加载时获取统计数据
            window.onload = function() {
                loadStats();
                loadExampleIds();
            };
            
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
            
            async function loadExampleIds() {
                try {
                    const response = await fetch('/patients?limit=10');
                    const data = await response.json();
                    
                    const exampleDiv = document.getElementById('exampleIds');
                    exampleDiv.innerHTML = '';
                    
                    if (data.patients && data.patients.length > 0) {
                        data.patients.forEach(pid => {
                            const tag = document.createElement('span');
                            tag.className = 'example-tag';
                            tag.textContent = pid;
                            tag.onclick = () => {
                                document.getElementById('patientId').value = pid;
                                predict();
                            };
                            exampleDiv.appendChild(tag);
                        });
                    } else {
                        // 默认示例
                        ['HN-CHUS-001', 'HN-CHUS-002-ok', 'HN-CHUS-003'].forEach(pid => {
                            const tag = document.createElement('span');
                            tag.className = 'example-tag';
                            tag.textContent = pid;
                            tag.onclick = () => {
                                document.getElementById('patientId').value = pid;
                                predict();
                            };
                            exampleDiv.appendChild(tag);
                        });
                    }
                } catch (error) {
                    console.error('加载示例ID失败:', error);
                }
            }
            
            async function predict() {
                const patientId = document.getElementById('patientId').value.trim();
                
                if (!patientId) {
                    alert('请输入病人ID');
                    return;
                }
                
                // 显示加载，隐藏结果和错误
                document.getElementById('loading').style.display = 'block';
                document.getElementById('result').style.display = 'none';
                document.getElementById('error').style.display = 'none';
                
                try {
                    const response = await fetch('/predict', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify({patient_id: patientId})
                    });
                    
                    // 检查响应状态
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
                    riskSpan.className = 'info-value ' + 
                        (data.risk_group === '高危' ? 'high-risk' : 'low-risk');
                    
                    // 显示各模型预测
                    const modelDiv = document.getElementById('modelDetails');
                    modelDiv.innerHTML = '';
                    
                    // 对预测结果排序
                    const sortedModels = Object.entries(data.predictions)
                        .sort((a, b) => b[1] - a[1]);
                    
                    sortedModels.forEach(([model, score]) => {
                        const item = document.createElement('div');
                        item.className = 'model-item';
                        // 简化模型名显示
                        let shortName = model.replace('model_', '').replace('.pth', '');
                        item.innerHTML = `
                            <span>模型 ${shortName}</span>
                            <span style="font-weight: bold; color: #667eea;">${score.toFixed(6)}</span>
                        `;
                        modelDiv.appendChild(item);
                    });
                    
                    document.getElementById('result').style.display = 'block';
                    
                } catch (error) {
                    console.error('预测错误:', error);
                    document.getElementById('error').style.display = 'block';
                    document.getElementById('error').textContent = '❌ 错误: ' + error.message;
                } finally {
                    document.getElementById('loading').style.display = 'none';
                }
            }
            
            // 按回车键查询
            document.getElementById('patientId').addEventListener('keypress', function(e) {
                if (e.key === 'Enter') {
                    predict();
                }
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
    print("✅ 所有准备工作完成！")
    print(f"📊 模型数量: {len(models)}")
    print(f"👥 病人数量: {len(patient_id_map)}")
    print(f"💻 设备: {device}")
    print("="*60)
    print("\n🌐 启动Web服务器...")
    print("🔗 访问地址: http://localhost:8000")
    print("   - 在浏览器中打开此地址")
    print("   - 从示例ID中选择或输入病人ID")
    print("="*60 + "\n")
    
    # 启动服务器
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )