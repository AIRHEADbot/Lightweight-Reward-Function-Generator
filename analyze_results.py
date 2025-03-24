# analyze_results.py
import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from stable_baselines3 import PPO
from stable_baselines3.common.results_plotter import load_results
from gymnasium.wrappers import RecordVideo
import torch

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="强化学习训练结果分析工具")
    parser.add_argument(
        "--log-dir", 
        type=str, 
        default="pong_saved_models", 
        help="训练输出目录路径"
    )
    return parser.parse_args()

def load_training_data(log_dir):
    """加载训练日志和模型"""
    # 加载监控数据
    monitor_path = os.path.join(log_dir, "training_log.monitor.csv")
    if not os.path.exists(monitor_path):
        raise FileNotFoundError(f"未找到监控文件: {monitor_path}")
    data = load_results(log_dir)
    
    # 加载最终模型
    model_path = os.path.join(log_dir, "final_model.zip")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"未找到模型文件: {model_path}")
    model = PPO.load(model_path)
    
    return data, model

def plot_training_metrics(data, save_dir):
    """绘制多指标训练曲线"""
    df = pd.DataFrame(data)
    plt.figure(figsize=(12, 8))
    
    # 奖励曲线
    plt.subplot(2, 2, 1)
    plt.plot(df["r"], alpha=0.3, label="Raw Reward")
    plt.plot(gaussian_filter1d(df["r"], sigma=2), label="Smoothed Reward")
    plt.xlabel("Timesteps")
    plt.ylabel("Reward")
    plt.legend()
    
    # 回合长度曲线
    plt.subplot(2, 2, 2)
    plt.plot(df["l"], label="Episode Length")
    plt.xlabel("Timesteps")
    plt.ylabel("Steps")
    
    # KL散度曲线（如果存在）
    if "train/approx_kl" in df.columns:
        plt.subplot(2, 2, 3)
        plt.plot(df["train/approx_kl"], label="KL Divergence")
        plt.xlabel("Timesteps")
        plt.ylabel("KL Divergence")
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "training_metrics.png"))
    plt.close()
    
    return os.path.join(save_dir, "training_metrics.png")

def plot_reward_curve(rewards, save_dir, title="Training Progress"):
    """绘制简单的奖励曲线"""
    plt.figure(figsize=(10, 6))
    plt.plot(rewards)
    plt.xlabel("Episodes")
    plt.ylabel("Reward")
    plt.title(title)
    save_path = os.path.join(save_dir, "training_curve.png")
    plt.savefig(save_path)
    plt.close()
    return save_path

def plot_action_heatmap(model, save_dir, obs_space_dim=3):
    """生成策略动作热力图"""
    try:
        # 生成网格点
        ball_y = np.linspace(0, 1, 20)
        paddle_y = np.linspace(0, 1, 20)
        actions = np.zeros((len(ball_y), len(paddle_y)))
        
        # 预测动作
        for i, by in enumerate(ball_y):
            for j, py in enumerate(paddle_y):
                if obs_space_dim == 3:
                    obs = np.array([0.5, by, py])
                else:
                    # 对于不同维度的观察空间，需要调整
                    obs = np.zeros(obs_space_dim)
                    if obs_space_dim > 3:
                        obs[0] = 0.5
                        obs[1] = by
                        obs[2] = py
                
                action, _ = model.predict(obs, deterministic=True)
                actions[i, j] = action
        
        # 绘制热力图
        plt.figure(figsize=(10, 8))
        plt.imshow(actions, cmap="viridis", origin="lower")
        plt.colorbar(label="Action")
        plt.xticks(np.arange(0, 20, 5), np.round(paddle_y[::5], 2))
        plt.yticks(np.arange(0, 20, 5), np.round(ball_y[::5], 2))
        plt.xlabel("Paddle Y Position")
        plt.ylabel("Ball Y Position")
        plt.title("Policy Action Distribution")
        save_path = os.path.join(save_dir, "action_heatmap.png")
        plt.savefig(save_path)
        plt.close()
        return save_path
    except Exception as e:
        print(f"生成策略热力图失败: {str(e)}")
        return None

def generate_video_demo(model, save_dir, env_creator=None):
    """生成演示视频"""
    try:
        from custom_pong_env import CustomPongEnv
        
        video_dir = os.path.join(save_dir, "demos")
        os.makedirs(video_dir, exist_ok=True)
        
        if env_creator:
            env = env_creator()
        else:
            env = CustomPongEnv()
            
        env = RecordVideo(env, video_folder=video_dir, name_prefix="demo")
        obs, _ = env.reset()
        terminated, truncated = False, False
        
        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, _ = env.step(action)
            
        env.close()
        return video_dir
    except Exception as e:
        print(f"生成演示视频失败: {str(e)}")
        return None

def export_statistics(data, save_dir):
    """导出统计表格"""
    df = pd.DataFrame(data)
    
    # 训练日志
    log_path = os.path.join(save_dir, "training_log.csv")
    df.to_csv(log_path, index=False)
    
    # 关键统计指标
    stats = {
        "Max Reward": df["r"].max(),
        "Mean Reward (Last 10%)": df["r"].iloc[-len(df)//10:].mean(),
        "Min Episode Length": df["l"].min(),
        "Max KL Divergence": df.get("train/approx_kl", pd.Series([np.nan])).max()
    }
    stats_path = os.path.join(save_dir, "summary_stats.csv")
    pd.DataFrame([stats]).to_csv(stats_path, index=False)
    
    return log_path, stats_path

def analyze_during_training(model, ep_info_buffer, save_dir, checkpoint_steps):
    """训练过程中的分析功能"""
    os.makedirs(save_dir, exist_ok=True)
    
    # 绘制简单的奖励曲线
    if ep_info_buffer:
        rewards_only = [ep_info["r"] for ep_info in ep_info_buffer]
        plot_path = plot_reward_curve(
            rewards_only, 
            save_dir, 
            f"Training Progress at {checkpoint_steps} steps"
        )
        print(f"训练曲线已保存到 {plot_path}")
    
    # 尝试生成策略热力图
    try:
        heatmap_path = plot_action_heatmap(model, save_dir)
        if heatmap_path:
            print(f"策略热力图已保存到 {heatmap_path}")
    except Exception as e:
        print(f"无法生成策略热力图: {str(e)}")

def main():
    args = parse_args()
    os.makedirs(args.log_dir, exist_ok=True)
    
    try:
        data, model = load_training_data(args.log_dir)
        
        # 生成可视化结果
        plot_training_metrics(data, args.log_dir)
        plot_action_heatmap(model, args.log_dir)
        generate_video_demo(model, args.log_dir)
        
        # 导出统计表格
        export_statistics(data, args.log_dir)
        print(f"分析结果已保存至: {args.log_dir}")
        
    except Exception as e:
        print(f"分析失败: {str(e)}")

if __name__ == "__main__":
    main()