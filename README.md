# DAS
处理分布式光纤声波传感信号，并通过卷积神经网络进行事件分类

## MATLAB 数据生成时域波形图

脚本：`scripts/mat_to_waterfall.py`

> 脚本文件名保持不变，但功能已改为输出时域信号图（非瀑布图）。

### 依赖

```bash
pip install numpy scipy matplotlib
```

### 用法示例

```bash
python scripts/mat_to_waterfall.py your_data.mat --sampling-rate 10000 --output-prefix result/signal
```

说明：
- `.mat` 中第 1 行按相位差处理，第 2 行按强度差处理；
- 原始数据为同一位置采集的一段时间序列，不进行二维重排；
- 每个 `.mat` 文件输出两张时域图：`*_phase.png` 和 `*_intensity.png`（横轴时间，纵轴幅值）。
