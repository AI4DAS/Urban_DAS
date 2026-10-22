# Highway vehicle-trace pipeline

这是整理出的主线：

```text
H5 → DAS 预处理 → F-K 方向分离 → 图片 → patch
   → YOLOv5 → 线段关联 → 轨迹 JSON → 可视化
```

原始目录没有被修改。DAS 读取和预处理沿用 `DasPrep.py` 的 H5、去趋势、去共模、Tukey taper 和低通实现；模型、阈值、通道范围、降采样、图片大小及输出结构来自原 `Highway_param.json`。

## 目录

```text
./
├── config/highway.json       # 数据、模型和 trace 1/2 参数
├── ckpt/best.pt              # YOLOv5 exp17 正式权重
├── highway_pipeline/         # 业务代码
├── yolov5/                   # 推理所需的精简 YOLOv5 代码
├── run.py                    # 使用入口
└── requirements.txt
```

相对路径均以仓库根目录为基准。默认使用：

- 数据：`Highway_2024_0510/data/data2/*.h5`
- 模型：`ckpt/best.pt`
- 输出：`Highway_2024_0510/output/0628_data2/trace_1` 或 `trace_2`

## 安装

建议使用原项目对应的 Python 3.11 环境：

```bash
cd Urban_DAS
python -m pip install -r requirements.txt
```

## 使用

完整运行 trace 1：

```bash
python run.py all --trace 1
```

完整运行 trace 2：

```bash
python run.py all --trace 2
```

也可以分阶段运行：

```bash
python run.py prepare --trace 1
python run.py trace --trace 1
```

`prepare`、`trace` 会清空各自对应的生成目录后重建，不会操作原始 H5、模型或旧源码目录。

## 输出

```text
trace_N/
├── pic/                  # org/pos/neg PNG 与轨迹 JSON
├── patches/              # 64×64、padding 后 96×96 的 patch
├── yolo/yolo/labels/     # YOLO 检测结果
└── predict_vis_result/   # 轨迹叠加图
```

JSON 保留原来的键名和相对坐标语义，包括 `h5 matadata`、`delet`、`associate pixel` 和 `trace_real_dic`，便于与旧结果对照。

当前本机 `Highway_2024_0510/data/data2` 为空；运行前需要先放入原始 H5 文件。
