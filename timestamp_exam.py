import argparse
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, TextIO, Tuple

import numpy as np
import pandas as pd

# 从当前目录导入 raw_reader
from raw_reader import (
    POSE_SOURCE_MAP,
    _align_clamp_timestamps,
    _ensure_file,
    _read_clamp_file,
    _read_pose_file,
    detect_layout,
    discover_sessions,
    extract_session_index,
)


@dataclass
class TimestampData:
    """存储单个数据源的时间戳数据"""
    name: str
    timestamps: np.ndarray
    ideal_freq: float  # 理想帧率 (Hz)
    actual_freq: Optional[float] = None  # 实机帧率 (Hz)


@dataclass
class SingleArmTimestampStats:
    """单臂时间戳统计"""
    rgb_clamp_diff: Dict[str, float]  # mean, max, min, first
    rgb_merge_diff: Dict[str, float]
    rgb_vive_diff: Dict[str, float]
    rgb_slam_diff: Dict[str, float]
    frequencies: Dict[str, Dict[str, float]]  # 每个数据源的理想和实机帧率


@dataclass
class DualArmTimestampStats:
    """双臂时间戳统计"""
    left_stats: SingleArmTimestampStats
    right_stats: SingleArmTimestampStats
    inter_arm_diff: Dict[str, Dict[str, float]]  # 双臂间各数据源的时间差统计


def read_rgb_timestamps(arm_root: str) -> np.ndarray:
    """读取RGB图像时间戳"""
    timestamps_path = os.path.join(arm_root, "RGB_Images", "timestamps.csv")
    _ensure_file(timestamps_path, "timestamps")
    timestamps_df = pd.read_csv(timestamps_path)
    if timestamps_df.empty:
        raise RuntimeError(f"{arm_root} timestamps.csv 为空")
    # 不进行降采样，读取所有时间戳
    return timestamps_df["aligned_stamp"].to_numpy()


def read_all_timestamp_sources(arm_root: str) -> Dict[str, TimestampData]:
    """读取单臂的所有数据源时间戳"""
    sources = {}
    
    # 读取RGB时间戳
    rgb_ts = read_rgb_timestamps(arm_root)
    sources["rgb"] = TimestampData(
        name="rgb",
        timestamps=rgb_ts,
        ideal_freq=30.0,  # 假设RGB相机30Hz
    )
    
    # 读取clamp数据 (200Hz)
    clamp_path = os.path.join(arm_root, "Clamp_Data", "clamp_data_tum.txt")
    if os.path.exists(clamp_path):
        clamp_df = _read_clamp_file(clamp_path)
        # 需要对齐时间戳
        merge_path = os.path.join(arm_root, "Merged_Trajectory", "merged_trajectory.txt")
        if os.path.exists(merge_path):
            merge_df = _read_pose_file(merge_path)
            clamp_df = _align_clamp_timestamps(clamp_df, merge_df)
        clamp_ts = clamp_df["timestamp"].to_numpy()
        sources["clamp"] = TimestampData(
            name="clamp",
            timestamps=clamp_ts,
            ideal_freq=200.0,
        )
    
    # 读取merge数据 (100Hz)
    merge_path = os.path.join(arm_root, "Merged_Trajectory", "merged_trajectory.txt")
    if os.path.exists(merge_path):
        merge_df = _read_pose_file(merge_path)
        merge_ts = merge_df["timestamp"].to_numpy()
        sources["merge"] = TimestampData(
            name="merge",
            timestamps=merge_ts,
            ideal_freq=100.0,
        )
    
    # 读取vive数据 (100Hz)
    vive_path = os.path.join(arm_root, "Vive_Poses", "vive_data_tum.txt")
    if os.path.exists(vive_path):
        vive_df = _read_pose_file(vive_path)
        vive_ts = vive_df["timestamp"].to_numpy()
        sources["vive"] = TimestampData(
            name="vive",
            timestamps=vive_ts,
            ideal_freq=100.0,
        )
    
    # 读取slam数据 (500Hz)
    slam_path = os.path.join(arm_root, "SLAM_Poses", "slam_raw.txt")
    if os.path.exists(slam_path):
        slam_df = _read_pose_file(slam_path)
        slam_ts = slam_df["timestamp"].to_numpy()
        sources["slam"] = TimestampData(
            name="slam",
            timestamps=slam_ts,
            ideal_freq=500.0,
        )
    
    return sources


def calculate_actual_frequency(timestamps: np.ndarray) -> Optional[float]:
    """计算实机帧率"""
    if len(timestamps) < 2:
        return None
    intervals = np.diff(timestamps)
    intervals = intervals[intervals > 0]  # 过滤掉非正数
    if len(intervals) == 0:
        return None
    mean_interval = np.mean(intervals)
    if mean_interval <= 0:
        return None
    return 1.0 / mean_interval


def calculate_time_diff_stats(
    reference_ts: np.ndarray,
    target_ts: np.ndarray,
) -> Dict[str, float]:
    """计算参考时间戳与目标时间戳最临近帧的时间差统计"""
    if len(target_ts) == 0:
        return {"mean": np.nan, "max": np.nan, "min": np.nan, "first": np.nan}
    
    diffs = []
    for ref_ts in reference_ts:
        nearest_idx = np.argmin(np.abs(target_ts - ref_ts))
        diff = abs(target_ts[nearest_idx] - ref_ts)
        diffs.append(diff)
    
    diffs = np.array(diffs)
    return {
        "mean": np.mean(diffs),
        "max": np.max(diffs),
        "min": np.min(diffs),
        "first": diffs[0] if len(diffs) > 0 else np.nan,
    }


def calculate_inter_arm_diff_stats(
    left_ts: np.ndarray,
    right_ts: np.ndarray,
) -> Dict[str, float]:
    """计算双臂间时间戳对齐的时间差统计（按索引对齐）"""
    min_len = min(len(left_ts), len(right_ts))
    if min_len == 0:
        return {"mean": np.nan, "max": np.nan, "min": np.nan, "first": np.nan}
    
    # 按索引对齐
    left_aligned = left_ts[:min_len]
    right_aligned = right_ts[:min_len]
    diffs = np.abs(left_aligned - right_aligned)
    
    return {
        "mean": np.mean(diffs),
        "max": np.max(diffs),
        "min": np.min(diffs),
        "first": diffs[0] if len(diffs) > 0 else np.nan,
    }


def analyze_single_arm(arm_root: str) -> SingleArmTimestampStats:
    """分析单臂的时间戳质量"""
    sources = read_all_timestamp_sources(arm_root)
    
    # 计算实机帧率
    for source in sources.values():
        source.actual_freq = calculate_actual_frequency(source.timestamps)
    
    rgb_ts = sources["rgb"].timestamps
    
    # 计算RGB与其他数据源的时间差
    stats = SingleArmTimestampStats(
        rgb_clamp_diff=calculate_time_diff_stats(rgb_ts, sources.get("clamp", TimestampData("", np.array([]), 0)).timestamps) if "clamp" in sources else {"mean": np.nan, "max": np.nan, "min": np.nan, "first": np.nan},
        rgb_merge_diff=calculate_time_diff_stats(rgb_ts, sources.get("merge", TimestampData("", np.array([]), 0)).timestamps) if "merge" in sources else {"mean": np.nan, "max": np.nan, "min": np.nan, "first": np.nan},
        rgb_vive_diff=calculate_time_diff_stats(rgb_ts, sources.get("vive", TimestampData("", np.array([]), 0)).timestamps) if "vive" in sources else {"mean": np.nan, "max": np.nan, "min": np.nan, "first": np.nan},
        rgb_slam_diff=calculate_time_diff_stats(rgb_ts, sources.get("slam", TimestampData("", np.array([]), 0)).timestamps) if "slam" in sources else {"mean": np.nan, "max": np.nan, "min": np.nan, "first": np.nan},
        frequencies={},
    )
    
    # 收集所有数据源的帧率信息
    for name, source in sources.items():
        stats.frequencies[name] = {
            "ideal": source.ideal_freq,
            "actual": source.actual_freq if source.actual_freq is not None else np.nan,
        }
    
    return stats


def analyze_dual_arm(left_root: str, right_root: str) -> DualArmTimestampStats:
    """分析双臂的时间戳质量"""
    left_stats = analyze_single_arm(left_root)
    right_stats = analyze_single_arm(right_root)
    
    # 读取双臂的所有数据源时间戳
    left_sources = read_all_timestamp_sources(left_root)
    right_sources = read_all_timestamp_sources(right_root)
    
    # 计算双臂间各数据源的时间差
    inter_arm_diff = {}
    for source_name in ["rgb", "clamp", "merge", "vive", "slam"]:
        if source_name in left_sources and source_name in right_sources:
            inter_arm_diff[source_name] = calculate_inter_arm_diff_stats(
                left_sources[source_name].timestamps,
                right_sources[source_name].timestamps,
            )
    
    return DualArmTimestampStats(
        left_stats=left_stats,
        right_stats=right_stats,
        inter_arm_diff=inter_arm_diff,
    )


def format_stats_output(stats: Dict[str, float], unit: str = "s") -> str:
    """格式化统计输出"""
    if unit == "s":
        multiplier = 1.0
        unit_str = "s"
    elif unit == "ms":
        multiplier = 1000.0
        unit_str = "ms"
    else:
        multiplier = 1.0
        unit_str = unit
    
    def format_val(val):
        if np.isnan(val):
            return "N/A"
        return f"{val * multiplier:.3f}{unit_str}"
    
    mean_val = stats["mean"]
    max_val = stats["max"]
    min_val = stats["min"]
    first_val = stats["first"]
    
    return f"平均值: {format_val(mean_val)}, 最大值: {format_val(max_val)}, 最小值: {format_val(min_val)}, 第一帧: {format_val(first_val)}"


def print_single_arm_report(session_name: str, stats: SingleArmTimestampStats, file: Optional[TextIO] = None):
    """打印单臂报告"""
    def output(text: str):
        print(text)
        if file:
            file.write(text + "\n")
    
    output(f"\n{'='*80}")
    output(f"Session: {session_name} (单臂)")
    output(f"{'='*80}")
    
    output("\n【时间戳差值统计 (RGB vs 其他数据源)】")
    output(f"RGB-Clamp:  {format_stats_output(stats.rgb_clamp_diff, 'ms')}")
    output(f"RGB-Merge:  {format_stats_output(stats.rgb_merge_diff, 'ms')}")
    output(f"RGB-Vive:   {format_stats_output(stats.rgb_vive_diff, 'ms')}")
    output(f"RGB-SLAM:   {format_stats_output(stats.rgb_slam_diff, 'ms')}")
    
    output("\n【帧率统计】")
    for source_name in ["rgb", "clamp", "merge", "vive", "slam"]:
        if source_name in stats.frequencies:
            freq_info = stats.frequencies[source_name]
            ideal = freq_info["ideal"]
            actual = freq_info["actual"]
            if not np.isnan(actual):
                output(f"{source_name.upper():8s}: 理想={ideal:6.1f}Hz, 实机={actual:6.2f}Hz")
            else:
                output(f"{source_name.upper():8s}: 理想={ideal:6.1f}Hz, 实机=N/A")


def print_dual_arm_report(session_name: str, stats: DualArmTimestampStats, file: Optional[TextIO] = None):
    """打印双臂报告"""
    def output(text: str):
        print(text)
        if file:
            file.write(text + "\n")
    
    output(f"\n{'='*80}")
    output(f"Session: {session_name} (双臂)")
    output(f"{'='*80}")
    
    output("\n【左臂 - 时间戳差值统计 (RGB vs 其他数据源)】")
    output(f"RGB-Clamp:  {format_stats_output(stats.left_stats.rgb_clamp_diff, 'ms')}")
    output(f"RGB-Merge:  {format_stats_output(stats.left_stats.rgb_merge_diff, 'ms')}")
    output(f"RGB-Vive:   {format_stats_output(stats.left_stats.rgb_vive_diff, 'ms')}")
    output(f"RGB-SLAM:   {format_stats_output(stats.left_stats.rgb_slam_diff, 'ms')}")
    
    output("\n【右臂 - 时间戳差值统计 (RGB vs 其他数据源)】")
    output(f"RGB-Clamp:  {format_stats_output(stats.right_stats.rgb_clamp_diff, 'ms')}")
    output(f"RGB-Merge:  {format_stats_output(stats.right_stats.rgb_merge_diff, 'ms')}")
    output(f"RGB-Vive:   {format_stats_output(stats.right_stats.rgb_vive_diff, 'ms')}")
    output(f"RGB-SLAM:   {format_stats_output(stats.right_stats.rgb_slam_diff, 'ms')}")
    
    output("\n【双臂间对齐时间差统计】")
    for source_name, diff_stats in stats.inter_arm_diff.items():
        output(f"{source_name.upper():8s}: {format_stats_output(diff_stats, 'ms')}")
    
    output("\n【左臂 - 帧率统计】")
    for source_name in ["rgb", "clamp", "merge", "vive", "slam"]:
        if source_name in stats.left_stats.frequencies:
            freq_info = stats.left_stats.frequencies[source_name]
            ideal = freq_info["ideal"]
            actual = freq_info["actual"]
            if not np.isnan(actual):
                output(f"{source_name.upper():8s}: 理想={ideal:6.1f}Hz, 实机={actual:6.2f}Hz")
            else:
                output(f"{source_name.upper():8s}: 理想={ideal:6.1f}Hz, 实机=N/A")
    
    output("\n【右臂 - 帧率统计】")
    for source_name in ["rgb", "clamp", "merge", "vive", "slam"]:
        if source_name in stats.right_stats.frequencies:
            freq_info = stats.right_stats.frequencies[source_name]
            ideal = freq_info["ideal"]
            actual = freq_info["actual"]
            if not np.isnan(actual):
                output(f"{source_name.upper():8s}: 理想={ideal:6.1f}Hz, 实机={actual:6.2f}Hz")
            else:
                output(f"{source_name.upper():8s}: 理想={ideal:6.1f}Hz, 实机=N/A")


def process_session(session_path: str, file: Optional[TextIO] = None) -> Tuple[str, bool, Optional[str]]:
    """处理单个session"""
    session_name = os.path.basename(session_path.rstrip(os.sep))
    try:
        mode, arms = detect_layout(session_path)
        if mode == "dual":
            stats = analyze_dual_arm(arms["left"], arms["right"])
            print_dual_arm_report(session_name, stats, file)
        else:
            arm_root = next(iter(arms.values()))
            stats = analyze_single_arm(arm_root)
            print_single_arm_report(session_name, stats, file)
        return session_name, True, None
    except Exception as exc:  # pylint: disable=broad-except
        error_msg = f"[FAIL] {session_name}: {str(exc)}"
        print(error_msg)
        if file:
            file.write(error_msg + "\n")
        return session_name, False, str(exc)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="评估原始数据的时间戳质量",
    )
    parser.add_argument(
        "--input_dirs",
        "-i",
        nargs="+",
        required=True,
        help="一个或多个 raw multi_sessions 目录",
    )
    args = parser.parse_args()
    
    # 按 input_dir 分组处理，为每个 multi_sessions 创建报告文件
    for input_dir in args.input_dirs:
        sessions = discover_sessions(input_dir)
        if not sessions:
            print(f"[WARNING] {input_dir} 未找到任何 session 目录")
            continue
        
        # 创建报告文件路径
        report_path = os.path.join(input_dir, "timestamp_quality_report.txt")
        
        print(f"[INFO] 处理 {input_dir}，共 {len(sessions)} 个 session")
        print(f"[INFO] 报告将保存到: {report_path}")
        
        # 打开文件并写入报告
        with open(report_path, "w", encoding="utf-8") as report_file:
            # 写入文件头
            report_file.write(f"时间戳质量评估报告\n")
            report_file.write(f"数据目录: {input_dir}\n")
            report_file.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            report_file.write(f"{'='*80}\n\n")
            
            # 处理每个 session
            for session_path in sessions:
                session_name, ok, message = process_session(session_path, report_file)
                if not ok:
                    # 错误信息已经在 process_session 中输出
                    pass
        
        print(f"[INFO] 报告已保存到: {report_path}")


if __name__ == "__main__":
    main()

