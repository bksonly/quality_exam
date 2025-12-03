## 1. 时间戳质量评估 (timestamp_exam.py)

评估原始数据的时间戳质量，计算RGB相机与其他数据源的时间差统计。

```bash
python3 timestamp_exam.py -i multi_sessions_20251201_213306/
```
**输出**：在每个 `multi_sessions` 目录下生成 `timestamp_quality_report.txt` 文件

## 2. 轨迹平滑度评估 (trajectory_exam.py)

评估SLAM、Vive、Merged三种轨迹的平滑程度，计算速度、加速度等指标的统计值。

```bash
python3 trajectory_exam.py -i multi_sessions_20251201_213306/ -o output_dir/
```

**输出**：
 默认输出路径在每个 `multi_sessions` 下
- CSV统计文件：`{multi_session_name}_trajectory_stats.csv`
- PNG轨迹图：每个session一个 `{session_name}_trajectory_plot.png`

## 处理多个目录

可以同时处理多个 `multi_sessions` 目录：

```bash
python3 timestamp_exam.py -i multi_sessions_1 multi_sessions_2 multi_sessions_3
python3 trajectory_exam.py -i multi_sessions_1 multi_sessions_2 multi_sessions_3
```

