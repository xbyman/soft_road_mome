# Road-MoME Demo

Road-MoME 道路异常检测离线演示平台。该项目将本地样本读取、单帧推理、四宫格可视化和结果导出整合为一个可直接演示的浏览器界面
## 项目定位


当前仓库主要提供：

- 基于 `demo_data/` 的本地样本读取与索引管理
- 基于 MoME 模型的单帧道路异常检测推理
- 基于 Dash 的浏览器可视化界面
- 推理结果导出为图片、JSON 和 CSV
- 面向 Windows 的 `PyInstaller + Waitress` 打包方案

## 功能特性

- 启动时自动校验配置、权重、样本索引和输出目录
- 从 `demo_data/index.json` 读取样本，并关联 `jpg` 与 `npz` 数据
- 支持选择样本、调节阈值、执行单帧检测
- 生成四宫格结果图，展示预测热力图与专家权重
- 支持导出：
  - 可视化图片 `outputs/visualizations/vis_<frame_id>.jpg`
  - 单帧结果日志 `outputs/logs/result_<frame_id>.json`
  - 汇总结果表 `outputs/logs/result_summary.csv`

## 项目结构

```text
road_mome_demo_v1/
├─ app.py                    # Dash 开发入口
├─ run_server.py             # Waitress 本地服务入口
├─ app_bootstrap.py          # 运行时初始化与启动校验
├─ build_exe.bat             # Windows 打包脚本
├─ config/                   # 配置文件
├─ core/                     # 核心实体、契约与异常定义
├─ data_access/              # index / 图片 / npz 读取
├─ demo_data/                # 演示样本
├─ runtime/                  # 推理、导出、日志、校验、可视化
├─ ui/                       # Dash 页面布局与回调
├─ models/                   # 模型定义
├─ tests/                    # 运行时校验相关测试
├─ docs/                     # 用户手册、模块说明、打包说明等文档
├─ weights/                  # 模型权重
├─ outputs/                  # 运行输出目录
├─ dist/                     # 打包产物
└─ build/                    # 打包中间产物
```

## 运行环境

- 操作系统：建议 Windows
- Python：建议 `3.10`
- 浏览器：Chrome / Edge 均可
- 关键资源需完整存在：
  - `config/`
  - `weights/road_mome_v12_best.pth`
  - `demo_data/index.json`
  - `demo_data/jpg/`
  - `demo_data/npz/`
  - `outputs/`

## 快速开始

### 1. 安装依赖

```powershell
pip install -r requirements_runtime.txt
pip install waitress
```

### 2. 开发模式启动

```powershell
python app.py
```

启动后访问：

```text
http://127.0.0.1:8050
```

### 3. 本地交付模式启动

```powershell
python run_server.py
```

`run_server.py` 会使用 `Waitress` 启动本地服务，更接近最终交付方式。

可选环境变量：

```powershell
$env:ROAD_MOME_PORT = "8060"
$env:ROAD_MOME_OPEN_BROWSER = "0"
python run_server.py
```

## 输入与输出

### 输入数据

- 样本索引：`demo_data/index.json`
- 原始图像：`demo_data/jpg/*.jpg`
- 推理特征：`demo_data/npz/*.npz`
- 模型权重：`weights/road_mome_v12_best.pth`

当前仓库内置了 5 个演示样本：

- `20230317074850.000`
- `20230317074850.200`
- `20230317074850.400`
- `20230317074850.600`
- `20230317074850.800`

### 输出结果

- 可视化结果图：`outputs/visualizations/`
- 结构化日志：`outputs/logs/result_*.json`
- 汇总 CSV：`outputs/logs/result_summary.csv`
- 运行日志：`outputs/logs/app.log`

## 打包说明

项目提供了 Windows 打包脚本：

```powershell
pip install pyinstaller waitress
.\build_exe.bat
```

打包完成后，产物位于：

```text
dist/road_mome_demo/
```

其中包含：

- `road_mome_demo.exe`
- `config/`
- `weights/`
- `demo_data/`
- `outputs/`
- `_internal/`

## 测试

当前仓库包含一组针对启动校验和样本校验异常路径的单元测试：

```powershell
python -m unittest tests.test_validators
```

## 文档

仓库内还包含更详细的中文说明文档：

- `docs/用户手册.md`
- `docs/打包与运行说明.md`
- `docs/测试说明.md`
- `docs/模块设计说明.md`
- `docs/版本说明.md`


##联系方式
邮箱：<3107848480@qq.com>
权重文件地址<https://huggingface.co/xingboyan/Road_Mome>
