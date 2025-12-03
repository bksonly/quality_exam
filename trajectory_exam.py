import argparse
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端，避免GUI依赖
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# 不需要显式导入 Axes3D，使用 projection='3d' 时会自动导入

# 从当前目录导入 raw_reader
from raw_reader import (
    POSE_SOURCE_MAP,
    _ensure_file,
    _read_pose_file,
    detect_layout,
    discover_sessions,
)


@dataclass
class TrajectoryData:
    """存储轨迹数据"""
    name: str
    timestamps: np.ndarray
    positions: np.ndarray  # (N, 3)
    quaternions: np.ndarray  # (N, 4) [qx, qy, qz, qw]


@dataclass
class SmoothnessMetrics:
    """平滑度指标"""
    displacement_norm: np.ndarray  # 前后两帧位置差的模（位移的模）
    linear_velocity_norm: np.ndarray
    linear_acceleration_norm: np.ndarray
    linear_jerk_norm: np.ndarray
    angular_velocity_norm: np.ndarray
    angular_acceleration_norm: np.ndarray
    angular_jerk_norm: np.ndarray


def read_trajectory_data(arm_root: str, source: str) -> Optional[TrajectoryData]:
    """读取指定数据源的轨迹数据"""
    if source not in POSE_SOURCE_MAP:
        return None
    
    subdir, filename = POSE_SOURCE_MAP[source]
    file_path = os.path.join(arm_root, subdir, filename)
    
    if not os.path.exists(file_path):
        return None
    
    df = _read_pose_file(file_path)
    if df.empty:
        return None
    
    timestamps = df["timestamp"].to_numpy()
    positions = df[["Pos X", "Pos Y", "Pos Z"]].to_numpy()
    quaternions = df[["Q_X", "Q_Y", "Q_Z", "Q_W"]].to_numpy()
    
    return TrajectoryData(
        name=source,
        timestamps=timestamps,
        positions=positions,
        quaternions=quaternions,
    )


def quaternion_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """四元数乘法: q1 * q2"""
    w1, x1, y1, z1 = q1[3], q1[0], q1[1], q1[2]
    w2, x2, y2, z2 = q2[3], q2[0], q2[1], q2[2]
    return np.array([
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
    ])


def quaternion_conjugate(q: np.ndarray) -> np.ndarray:
    """四元数共轭: q* = [-qx, -qy, -qz, qw]"""
    return np.array([-q[0], -q[1], -q[2], q[3]])


def quaternion_to_angular_velocity(q1: np.ndarray, q2: np.ndarray, dt: float) -> np.ndarray:
    """
    计算从四元数 q1 到 q2 的角速度
    使用公式: q_diff = q2 * q1^(-1), 然后从 q_diff 提取角速度
    """
    if dt < 1e-10:
        return np.zeros(3)
    
    # 归一化四元数
    q1 = q1 / np.linalg.norm(q1)
    q2 = q2 / np.linalg.norm(q2)
    
    # 计算相对旋转: q_diff = q2 * q1^(-1) = q2 * q1*
    q1_conj = quaternion_conjugate(q1)
    q_diff = quaternion_multiply(q2, q1_conj)
    
    # 归一化
    q_diff = q_diff / np.linalg.norm(q_diff)
    
    # 从四元数提取角速度
    # 对于单位四元数 q = [qx, qy, qz, qw]，如果表示小旋转
    # 角速度 ω ≈ 2 * [qx, qy, qz] / dt (当 qw ≈ 1)
    # 对于一般情况，使用轴角表示
    qw = q_diff[3]
    qv = q_diff[:3]
    
    # 计算旋转角度
    angle = 2 * np.arccos(np.clip(abs(qw), 0, 1))
    
    if angle < 1e-6:
        # 小角度近似
        omega = 2 * qv / dt
    else:
        # 使用轴角表示: ω = (angle / dt) * axis
        axis_norm = np.linalg.norm(qv)
        if axis_norm > 1e-10:
            axis = qv / axis_norm
            omega = axis * angle / dt
        else:
            omega = np.zeros(3)
    
    return omega


def compute_derivatives(values: np.ndarray, timestamps: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    计算一阶、二阶、三阶导数
    返回: (velocity, acceleration, jerk)
    """
    if len(values) < 2:
        if values.ndim == 1:
            return np.zeros_like(values), np.zeros_like(values), np.zeros_like(values)
        else:
            n = len(values)
            return np.zeros((n, values.shape[1])), np.zeros((n, values.shape[1])), np.zeros((n, values.shape[1]))
    
    # 使用数值微分
    dt = np.diff(timestamps)
    
    # 一阶导数（速度）- 使用中心差分
    if values.ndim == 1:
        velocity = np.zeros_like(values)
        velocity[0] = (values[1] - values[0]) / dt[0] if len(dt) > 0 else 0
        for i in range(1, len(values) - 1):
            velocity[i] = (values[i+1] - values[i-1]) / (dt[i-1] + dt[i])
        if len(values) > 1:
            velocity[-1] = (values[-1] - values[-2]) / dt[-1]
    else:
        velocity = np.zeros_like(values)
        velocity[0] = (values[1] - values[0]) / dt[0] if len(dt) > 0 else 0
        for i in range(1, len(values) - 1):
            velocity[i] = (values[i+1] - values[i-1]) / (dt[i-1] + dt[i])
        if len(values) > 1:
            velocity[-1] = (values[-1] - values[-2]) / dt[-1]
    
    # 二阶导数（加速度）
    if velocity.ndim == 1:
        acceleration = np.zeros_like(velocity)
        if len(velocity) > 1:
            acceleration[0] = (velocity[1] - velocity[0]) / dt[0] if len(dt) > 0 else 0
            for i in range(1, len(velocity) - 1):
                acceleration[i] = (velocity[i+1] - velocity[i-1]) / (dt[i-1] + dt[i])
            acceleration[-1] = (velocity[-1] - velocity[-2]) / dt[-1]
    else:
        acceleration = np.zeros_like(velocity)
        if len(velocity) > 1:
            acceleration[0] = (velocity[1] - velocity[0]) / dt[0] if len(dt) > 0 else 0
            for i in range(1, len(velocity) - 1):
                acceleration[i] = (velocity[i+1] - velocity[i-1]) / (dt[i-1] + dt[i])
            acceleration[-1] = (velocity[-1] - velocity[-2]) / dt[-1]
    
    # 三阶导数（加加速度/jerk）
    if acceleration.ndim == 1:
        jerk = np.zeros_like(acceleration)
        if len(acceleration) > 1:
            jerk[0] = (acceleration[1] - acceleration[0]) / dt[0] if len(dt) > 0 else 0
            for i in range(1, len(acceleration) - 1):
                jerk[i] = (acceleration[i+1] - acceleration[i-1]) / (dt[i-1] + dt[i])
            jerk[-1] = (acceleration[-1] - acceleration[-2]) / dt[-1]
    else:
        jerk = np.zeros_like(acceleration)
        if len(acceleration) > 1:
            jerk[0] = (acceleration[1] - acceleration[0]) / dt[0] if len(dt) > 0 else 0
            for i in range(1, len(acceleration) - 1):
                jerk[i] = (acceleration[i+1] - acceleration[i-1]) / (dt[i-1] + dt[i])
            jerk[-1] = (acceleration[-1] - acceleration[-2]) / dt[-1]
    
    return velocity, acceleration, jerk


def compute_angular_derivatives(quaternions: np.ndarray, timestamps: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    计算角速度、角加速度、角加加速度
    返回: (angular_velocity, angular_acceleration, angular_jerk)
    """
    if len(quaternions) < 2:
        n = len(quaternions)
        return np.zeros((n, 3)), np.zeros((n, 3)), np.zeros((n, 3))
    
    # 归一化四元数
    norms = np.linalg.norm(quaternions, axis=1, keepdims=True)
    quaternions = quaternions / np.where(norms > 1e-10, norms, 1.0)
    
    # 计算角速度
    angular_velocities = []
    dt = np.diff(timestamps)
    dt = np.concatenate([[dt[0] if len(dt) > 0 else 1e-3], dt])
    
    for i in range(len(quaternions)):
        if i == 0:
            # 第一个点使用前向差分
            if len(quaternions) > 1:
                omega = quaternion_to_angular_velocity(quaternions[0], quaternions[1], dt[0])
            else:
                omega = np.zeros(3)
        else:
            # 使用后向差分
            omega = quaternion_to_angular_velocity(quaternions[i-1], quaternions[i], dt[i])
        angular_velocities.append(omega)
    
    angular_velocities = np.array(angular_velocities)
    
    # 计算角加速度和角加加速度
    _, angular_acceleration, angular_jerk = compute_derivatives(angular_velocities, timestamps)
    
    return angular_velocities, angular_acceleration, angular_jerk


def compute_smoothness_metrics(trajectory: TrajectoryData) -> SmoothnessMetrics:
    """计算平滑度指标"""
    # 计算前后两帧位置差的模（位移的模）
    if len(trajectory.positions) < 2:
        displacement_norm = np.array([0.0] * len(trajectory.positions))
    else:
        displacement = np.diff(trajectory.positions, axis=0)
        displacement_norm_diff = np.linalg.norm(displacement, axis=1)
        # 第一帧没有位移（设为0），后续帧使用计算出的位移模
        displacement_norm = np.concatenate([[0.0], displacement_norm_diff])
    
    # 计算线速度、加速度、加加速度
    linear_velocity, linear_acceleration, linear_jerk = compute_derivatives(
        trajectory.positions, trajectory.timestamps
    )
    
    # 计算角速度、角加速度、角加加速度
    angular_velocity, angular_acceleration, angular_jerk = compute_angular_derivatives(
        trajectory.quaternions, trajectory.timestamps
    )
    
    return SmoothnessMetrics(
        displacement_norm=displacement_norm,
        linear_velocity_norm=np.linalg.norm(linear_velocity, axis=1),
        linear_acceleration_norm=np.linalg.norm(linear_acceleration, axis=1),
        linear_jerk_norm=np.linalg.norm(linear_jerk, axis=1),
        angular_velocity_norm=np.linalg.norm(angular_velocity, axis=1),
        angular_acceleration_norm=np.linalg.norm(angular_acceleration, axis=1),
        angular_jerk_norm=np.linalg.norm(angular_jerk, axis=1),
    )


def analyze_trajectories(arm_root: str) -> Dict[str, SmoothnessMetrics]:
    """分析所有可用的轨迹数据"""
    results = {}
    
    for source in ["slam", "vive", "merged"]:
        trajectory = read_trajectory_data(arm_root, source)
        if trajectory is not None:
            metrics = compute_smoothness_metrics(trajectory)
            results[source] = metrics
    
    return results


def create_statistics_table(results: Dict[str, SmoothnessMetrics]) -> pd.DataFrame:
    """创建统计表格"""
    metrics_names = [
        "位移模",
        "线速度模",
        "线加速度模",
        "线加加速度模",
        "角速度模",
        "角加速度模",
        "角加加速度模",
    ]
    
    # 创建表格数据 - 使用多级列索引
    table_data = []
    
    for metric_name in metrics_names:
        row = {"指标": metric_name}
        
        for source in ["slam", "vive", "merged"]:
            if source in results:
                metrics = results[source]
                if metric_name == "位移模":
                    values = metrics.displacement_norm
                elif metric_name == "线速度模":
                    values = metrics.linear_velocity_norm
                elif metric_name == "线加速度模":
                    values = metrics.linear_acceleration_norm
                elif metric_name == "线加加速度模":
                    values = metrics.linear_jerk_norm
                elif metric_name == "角速度模":
                    values = metrics.angular_velocity_norm
                elif metric_name == "角加速度模":
                    values = metrics.angular_acceleration_norm
                elif metric_name == "角加加速度模":
                    values = metrics.angular_jerk_norm
                else:
                    values = np.array([])
                
                if len(values) > 0:
                    max_val = np.max(values)
                    mean_val = np.mean(values)
                    # 使用分号分隔，便于CSV读取
                    row[f"{source}_最大值"] = max_val
                    row[f"{source}_平均值"] = mean_val
                else:
                    row[f"{source}_最大值"] = np.nan
                    row[f"{source}_平均值"] = np.nan
            else:
                row[f"{source}_最大值"] = np.nan
                row[f"{source}_平均值"] = np.nan
        
        table_data.append(row)
    
    df = pd.DataFrame(table_data)
    return df


def plot_trajectories(results: Dict[str, SmoothnessMetrics], arm_root: str, output_path: str, title: str = "3D Trajectory"):
    """绘制3D轨迹图"""
    # 读取轨迹数据用于绘图
    trajectories = {}
    for source in ["slam", "vive", "merged"]:
        traj = read_trajectory_data(arm_root, source)
        if traj is not None:
            trajectories[source] = traj
    
    if not trajectories:
        print(f"[WARNING] 没有可用的轨迹数据用于绘图")
        return None
    
    # 创建图形 - 只绘制3D图
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    colors = {"slam": "r", "vive": "g", "merged": "b"}
    for source, traj in trajectories.items():
        ax.plot(
            traj.positions[:, 0],
            traj.positions[:, 1],
            traj.positions[:, 2],
            color=colors.get(source, "k"),
            label=source.upper(),
            alpha=0.7,
            linewidth=1.5,
        )
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title(title)
    ax.legend()
    ax.grid(True)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[INFO] 轨迹图已保存到: {output_path}")
    return fig


def plot_dual_arm_trajectories(left_results: Dict[str, SmoothnessMetrics], left_arm_root: str,
                                right_results: Dict[str, SmoothnessMetrics], right_arm_root: str,
                                output_path: str):
    """绘制双臂的3D轨迹图（左右两个子图）"""
    # 读取左右臂的轨迹数据
    left_trajectories = {}
    right_trajectories = {}
    
    for source in ["slam", "vive", "merged"]:
        left_traj = read_trajectory_data(left_arm_root, source)
        if left_traj is not None:
            left_trajectories[source] = left_traj
        
        right_traj = read_trajectory_data(right_arm_root, source)
        if right_traj is not None:
            right_trajectories[source] = right_traj
    
    if not left_trajectories and not right_trajectories:
        print(f"[WARNING] 没有可用的轨迹数据用于绘图")
        return
    
    # 创建图形 - 左右两个子图
    fig = plt.figure(figsize=(24, 10))
    
    colors = {"slam": "r", "vive": "g", "merged": "b"}
    
    # 左臂子图
    if left_trajectories:
        ax1 = fig.add_subplot(121, projection='3d')
        for source, traj in left_trajectories.items():
            ax1.plot(
                traj.positions[:, 0],
                traj.positions[:, 1],
                traj.positions[:, 2],
                color=colors.get(source, "k"),
                label=source.upper(),
                alpha=0.7,
                linewidth=1.5,
            )
        ax1.set_xlabel("X (m)")
        ax1.set_ylabel("Y (m)")
        ax1.set_zlabel("Z (m)")
        ax1.set_title("Left Arm - 3D Trajectory")
        ax1.legend()
        ax1.grid(True)
    
    # 右臂子图
    if right_trajectories:
        ax2 = fig.add_subplot(122, projection='3d')
        for source, traj in right_trajectories.items():
            ax2.plot(
                traj.positions[:, 0],
                traj.positions[:, 1],
                traj.positions[:, 2],
                color=colors.get(source, "k"),
                label=source.upper(),
                alpha=0.7,
                linewidth=1.5,
            )
        ax2.set_xlabel("X (m)")
        ax2.set_ylabel("Y (m)")
        ax2.set_zlabel("Z (m)")
        ax2.set_title("Right Arm - 3D Trajectory")
        ax2.legend()
        ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[INFO] 轨迹图已保存到: {output_path}")


def process_session(session_path: str, output_dir: str) -> Tuple[str, List[Dict], bool, Optional[str]]:
    """
    处理单个session
    返回: (session_name, stats_list, success, error_message)
    stats_list 包含每个arm的统计数据和DataFrame
    """
    session_name = os.path.basename(session_path.rstrip(os.sep))
    stats_list = []
    
    try:
        mode, arms = detect_layout(session_path)
        
        # 确定要处理的arm路径
        if mode == "dual":
            # 双臂模式：处理左臂和右臂
            arm_roots = [arms["left"], arms["right"]]
            arm_names = ["left", "right"]
        else:
            # 单臂模式
            arm_roots = [next(iter(arms.values()))]
            arm_names = ["single"]
        
        # 如果是双臂模式，收集所有arm的数据后再统一绘制
        if mode == "dual":
            left_results = analyze_trajectories(arm_roots[0])
            right_results = analyze_trajectories(arm_roots[1])
            
            # 保存统计数据
            if left_results:
                left_stats_df = create_statistics_table(left_results)
                stats_list.append({
                    "session": session_name,
                    "arm": "left",
                    "results": left_results,
                    "dataframe": left_stats_df,
                })
            
            if right_results:
                right_stats_df = create_statistics_table(right_results)
                stats_list.append({
                    "session": session_name,
                    "arm": "right",
                    "results": right_results,
                    "dataframe": right_stats_df,
                })
            
            # 绘制双臂轨迹图（左右两个子图）
            if left_results or right_results:
                plot_filename = f"{session_name}_trajectory_plot.png"
                plot_path = os.path.join(output_dir, plot_filename)
                plot_dual_arm_trajectories(
                    left_results, arm_roots[0],
                    right_results, arm_roots[1],
                    plot_path
                )
        else:
            # 单臂模式，单独处理
            for arm_root, arm_name in zip(arm_roots, arm_names):
                # 分析轨迹
                results = analyze_trajectories(arm_root)
                
                if not results:
                    print(f"[WARNING] {session_name} ({arm_name}) 没有可用的轨迹数据")
                    continue
                
                # 创建统计表格
                stats_df = create_statistics_table(results)
                
                # 保存统计数据
                stats_list.append({
                    "session": session_name,
                    "arm": arm_name,
                    "results": results,
                    "dataframe": stats_df,
                })
                
                # 绘制并保存轨迹图
                plot_filename = f"{session_name}_{arm_name}_trajectory_plot.png"
                plot_path = os.path.join(output_dir, plot_filename)
                plot_trajectories(results, arm_root, plot_path)
        
        return session_name, stats_list, True, None
    except Exception as exc:  # pylint: disable=broad-except
        return session_name, [], False, str(exc)


def merge_statistics_tables(stats_list: List[Dict]) -> pd.DataFrame:
    """合并多个session的统计表格"""
    if not stats_list:
        return pd.DataFrame()
    
    # 收集所有数据
    merged_dfs = []
    
    for stat in stats_list:
        session_name = stat["session"]
        arm_name = stat["arm"]
        df = stat["dataframe"].copy()
        
        # 为DataFrame添加session和arm信息
        df.insert(0, "Arm", arm_name)
        df.insert(0, "Session", session_name)
        
        merged_dfs.append(df)
    
    if not merged_dfs:
        return pd.DataFrame()
    
    # 合并所有DataFrame
    merged_df = pd.concat(merged_dfs, ignore_index=True)
    return merged_df


def main() -> None:
    parser = argparse.ArgumentParser(
        description="评估轨迹的平滑程度",
    )
    parser.add_argument(
        "--input_dirs",
        "-i",
        nargs="+",
        required=True,
        help="一个或多个 raw multi_sessions 目录",
    )
    parser.add_argument(
        "--output_dir",
        "-o",
        default=None,
        help="输出目录（默认：每个multi_sessions目录下）",
    )
    args = parser.parse_args()
    
    # 按 input_dir 分组处理
    for input_dir in args.input_dirs:
        sessions = discover_sessions(input_dir)
        if not sessions:
            print(f"[WARNING] {input_dir} 未找到任何 session 目录")
            continue
        
        # 确定输出目录
        if args.output_dir:
            output_dir = args.output_dir
        else:
            output_dir = input_dir
        
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"[INFO] 处理 {input_dir}，共 {len(sessions)} 个 session")
        print(f"[INFO] 输出目录: {output_dir}")
        
        # 收集所有session的统计数据
        all_stats = []
        
        # 处理每个 session
        for session_path in sessions:
            session_name, stats_list, ok, message = process_session(session_path, output_dir)
            if not ok:
                print(f"[FAIL] {session_name}: {message}")
            else:
                all_stats.extend(stats_list)
        
        # 合并所有session的统计数据并保存为一个CSV文件
        if all_stats:
            merged_df = merge_statistics_tables(all_stats)
            if not merged_df.empty:
                # 获取multi_session目录名
                multi_session_name = os.path.basename(input_dir.rstrip(os.sep))
                csv_filename = f"{multi_session_name}_trajectory_stats.csv"
                csv_path = os.path.join(output_dir, csv_filename)
                merged_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
                print(f"[INFO] 合并的统计表格已保存到: {csv_path}")
            else:
                print(f"[WARNING] 没有可用的统计数据")
        else:
            print(f"[WARNING] 没有收集到任何统计数据")


if __name__ == "__main__":
    main()

