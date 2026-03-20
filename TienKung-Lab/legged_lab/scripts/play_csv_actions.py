#!/usr/bin/env python3
# Copyright (c) 2025-2026, The TienKung-Lab Project Developers.
# All rights reserved.
# Modifications are licensed under the BSD-3-Clause license.

"""
MuJoCo G1 机器人 CSV 动作回放脚本

该脚本用于加载 CSV 格式的动作数据并在 G1 机器人 MuJoCo 模型上播放。

CSV 格式要求:
- 每一行代表一帧
- 每行包含 29 个关节位置值（对应 G1 的 29 DOF）
- 使用逗号分隔

使用方法:
    python legged_lab/scripts/play_csv_actions.py --actions path/to/actions.csv

参数:
    --model: MuJoCo XML 模型路径 (默认: g1_29dof_rev_1_0_daf.xml)
    --actions: CSV 动作文件路径 (必需)
    --freq: 控制频率 Hz (默认: 50)
    --kp: PD 控制器 P 增益 (默认: 100.0)
    --kd: PD 控制器 D 增益 (默认: 1.0)

示例:
    python legged_lab/scripts/play_csv_actions.py \\
        --actions legged_lab/envs/g1/datasets/run_subject1.csv \\
        --freq 50
"""

import os
import sys
import argparse
import time
import numpy as np
import mujoco
import mujoco.viewer


# 默认配置
NUM_JOINTS = 29  # G1 机器人总关节数


def load_csv_actions(csv_file):
    """
    加载 CSV 格式的动作数据

    Args:
        csv_file: CSV 文件路径

    Returns:
        numpy.ndarray: 动作数据数组 [num_frames, NUM_JOINTS]

    Raises:
        ValueError: 如果 CSV 列数不等于 NUM_JOINTS
    """
    print(f"[INFO] 正在加载 CSV 动作数据: {csv_file}")
    data = np.loadtxt(csv_file, delimiter=',')

    if data.ndim == 1:
        # 单行数据，扩展为 2D
        data = data.reshape(1, -1)

    if data.shape[1] != NUM_JOINTS:
        raise ValueError(
            f"CSV 文件列数应为 {NUM_JOINTS}，但检测到 {data.shape[1]} 列\n"
            f"请确保 CSV 文件每行包含 {NUM_JOINTS} 个关节位置值"
        )

    print(f"[INFO] 成功加载 {data.shape[0]} 帧动作数据")
    return data


def play_actions_in_mujoco(model_path, actions_data, control_freq, kp, kd):
    """
    在 MuJoCo 中播放动作数据

    Args:
        model_path: MuJoCo XML 模型路径
        actions_data: 动作数据数组 [num_frames, NUM_JOINTS]
        control_freq: 控制频率 (Hz)
        kp: PD 控制器 P 增益
        kd: PD 控制器 D 增益
    """
    print(f"[INFO] 加载 MuJoCo 模型: {model_path}")
    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)

    print(f"[INFO] 启动 MuJoCo 仿真窗口")
    viewer = mujoco.viewer.launch_passive(model, data)

    num_frames = actions_data.shape[0]
    print(f"[INFO] 开始播放 {num_frames} 帧动作")
    print(f"[INFO] 控制频率: {control_freq} Hz")
    print(f"[INFO] PD 增益: Kp={kp}, Kd={kd}")
    print(f"[INFO] 按窗口关闭按钮以停止播放")
    print("-" * 60)

    frame_idx = 0
    t_start = time.time()

    while viewer.is_running() and frame_idx < num_frames:
        # 获取本帧目标关节位置
        qpos_target = actions_data[frame_idx]

        # 读取当前实际关节位置和速度
        # MuJoCo qpos: [base_pos(3) + base_quat(4) + joint_pos(29)]
        # MuJoCo qvel: [base_lin_vel(3) + base_ang_vel(3) + joint_vel(29)]
        qpos_current = data.qpos[7:7 + NUM_JOINTS]
        qvel_current = data.qvel[6:6 + NUM_JOINTS]

        # PD 控制计算关节力矩
        tau = kp * (qpos_target - qpos_current) - kd * qvel_current

        # 发送控制指令
        data.ctrl[:NUM_JOINTS] = tau

        # 推进仿真一步
        mujoco.mj_step(model, data)
        viewer.sync()

        # 控制循环频率
        elapsed_time = time.time() - t_start
        target_time = (frame_idx + 1) / control_freq
        sleep_time = target_time - elapsed_time
        if sleep_time > 0:
            time.sleep(sleep_time)

        # 每 100 帧打印进度
        if (frame_idx + 1) % 100 == 0:
            progress = (frame_idx + 1) / num_frames * 100
            print(f"[INFO] 进度: {frame_idx + 1}/{num_frames} ({progress:.1f}%)")

        frame_idx += 1

    print("-" * 60)
    if frame_idx >= num_frames:
        print(f"[INFO] 动作播放完成！")
    else:
        print(f"[INFO] 播放已停止（窗口关闭）")

    viewer.close()


def main():
    # 获取脚本所在目录（用于构建默认模型路径）
    script_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
    default_model = os.path.join(
        script_dir,
        "legged_lab/assets/unitree/g1/mjcf/g1_29dof_rev_1_0_daf.xml"
    )

    parser = argparse.ArgumentParser(
        description="MuJoCo G1 机器人 CSV 动作回放工具",
        formatter_class=argparse.RawTextHelpFormatter
    )

    parser.add_argument(
        "--model",
        type=str,
        default=default_model,
        help="MuJoCo XML 模型路径\n默认: g1_29dof_rev_1_0_daf.xml"
    )

    parser.add_argument(
        "--actions",
        type=str,
        required=True,
        help="CSV 动作文件路径 (必需)\n每行 29 个关节位置值，逗号分隔"
    )

    parser.add_argument(
        "--freq",
        type=int,
        default=50,
        help="控制频率 (Hz)\n默认: 50"
    )

    parser.add_argument(
        "--kp",
        type=float,
        default=100.0,
        help="PD 控制器 P 增益\n默认: 100.0"
    )

    parser.add_argument(
        "--kd",
        type=float,
        default=1.0,
        help="PD 控制器 D 增益\n默认: 1.0"
    )

    args = parser.parse_args()

    # 检查文件是否存在
    if not os.path.isfile(args.model):
        print(f"[ERROR] MuJoCo 模型文件不存在: {args.model}")
        sys.exit(1)

    if not os.path.isfile(args.actions):
        print(f"[ERROR] CSV 动作文件不存在: {args.actions}")
        sys.exit(1)

    print("=" * 60)
    print("MuJoCo G1 机器人 CSV 动作回放")
    print("=" * 60)

    try:
        # 加载动作数据
        actions_data = load_csv_actions(args.actions)

        # 在 MuJoCo 中播放
        play_actions_in_mujoco(
            model_path=args.model,
            actions_data=actions_data,
            control_freq=args.freq,
            kp=args.kp,
            kd=args.kd
        )

    except KeyboardInterrupt:
        print("\n[INFO] 用户中断播放")
    except Exception as e:
        print(f"\n[ERROR] 播放过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    print("=" * 60)


if __name__ == "__main__":
    main()
