import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch
from dotenv import load_dotenv
from typing import Any, Dict
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.results_plotter import load_results
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from custom_pong_env import CustomPongEnv
from generate_reward import generate_reward_function, validate_reward

# 导入分析模块
import analyze_results as analyzer

# 限制PyTorch线程
torch.set_num_threads(1)

# 强制使用无交互式后端
plt.switch_backend('Agg')

# 检查是否有可用的GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# 指定 .env 文件路径
script_dir = os.path.dirname(os.path.abspath(__file__))
env_path = os.path.join(script_dir, ".env")
load_dotenv(env_path)

class Config:
    TOTAL_TIMESTEPS: int = 100_000
    SAVE_FREQ: int = 20_000
    SAVE_DIR: str = os.path.join(script_dir, "pong_saved_models")  # 改为相对路径
    POLICY: str = "MlpPolicy"
    N_ENVS: int = 1  # 并行环境数量，根据GPU资源调整
    VISUALIZATION_FREQ: int = 10_000  # 可视化频率

config = Config()

class SaveModelCallback(BaseCallback):
    def __init__(self, save_freq: int, save_path: str, viz_freq: int, verbose: int = 0):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path
        self.viz_freq = viz_freq
        self.last_viz_time = 0
        
    def _init_callback(self) -> None:
        os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        # 保存模型
        if self.n_calls % self.save_freq == 0:
            model_path = os.path.join(self.save_path, f"model_{self.n_calls}")
            self.model.save(model_path)
            print(f"已保存模型快照: {model_path}")
            
        # 在训练过程中进行可视化分析
        if self.n_calls % self.viz_freq == 0 and self.n_calls > self.last_viz_time:
            self.last_viz_time = self.n_calls
            checkpoint_dir = os.path.join(self.save_path, f"checkpoint_{self.n_calls}")
            os.makedirs(checkpoint_dir, exist_ok=True)
            
            # 使用分析模块生成可视化
            analyzer.analyze_during_training(
                self.model, 
                self.model.ep_info_buffer, 
                checkpoint_dir,
                self.n_calls
            )
            
            print(f"已生成训练进度可视化: {checkpoint_dir}")
            
        return True

class RewardWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env, reward_func: Any):
        super().__init__(env)
        self.reward_func = reward_func

    def step(self, action: int) -> tuple:
        obs, _, terminated, truncated, info = self.env.step(action)
        reward = self.reward_func(obs)
        return obs, reward, terminated, truncated, info

def make_env(reward_func, rank=0):
    """
    创建一个环境的工厂函数
    """
    def _init():
        env = CustomPongEnv()
        env = RewardWrapper(env, reward_func)
        env = Monitor(env, os.path.join(config.SAVE_DIR, f"env_{rank}"))  # 使用Monitor记录数据
        return env
    return _init

def main():
    # 获取API密钥
    deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")
    if not deepseek_api_key:
        print("未找到DEEPSEEK_API_KEY环境变量")
        raise ValueError("API密钥未设置，请在.env文件中设置DEEPSEEK_API_KEY")

    # 生成奖励函数
    reward_func = None
    for retry in range(3):
        try:
            code = generate_reward_function(deepseek_api_key)
            reward_func = validate_reward(code)
            if reward_func:
                print("成功生成奖励函数")
                break
        except Exception as e:
            print(f"生成失败 (尝试 {retry+1}/3): {str(e)}")
    
    if reward_func is None:
        print("无法生成有效的奖励函数，使用默认奖励函数")
        code = """def calculate_reward(state):
            ball_x = state[0]
            ball_y = state[1]
            paddle_y = state[2]
            vertical_distance = abs(paddle_y - ball_y)
            x_factor = ball_x ** 2
            reward = -vertical_distance * (1 + 5 * x_factor)
            return float(reward)"""
        reward_func = validate_reward(code)
        if not reward_func:
            raise RuntimeError("默认奖励函数也失败了，无法继续")

    # 创建并行环境
    env_fns = [make_env(reward_func, i) for i in range(config.N_ENVS)]
    env = DummyVecEnv(env_fns)
    
    # 确保保存目录存在
    os.makedirs(config.SAVE_DIR, exist_ok=True)
    
    # 设置GPU加速
    policy_kwargs = dict(
        activation_fn=torch.nn.ReLU,
        net_arch=dict(pi=[64, 64], vf=[64, 64])
    )
    
    # 创建PPO模型并指定设备
    model = PPO(
        config.POLICY, 
        env, 
        verbose=1,
        policy_kwargs=policy_kwargs,
        device=device
    )
    
    # 设置回调，包含可视化功能
    callback = SaveModelCallback(
        save_freq=config.SAVE_FREQ, 
        save_path=config.SAVE_DIR,
        viz_freq=config.VISUALIZATION_FREQ
    )
    
    print(f"开始训练 - 使用设备: {model.device}")
    
    # 训练模型
    model.learn(total_timesteps=config.TOTAL_TIMESTEPS, callback=callback)
    final_model_path = os.path.join(config.SAVE_DIR, "final_model")
    model.save(final_model_path)
    
    print(f"训练完成，最终模型已保存至: {final_model_path}")
    
    # 训练结束后进行全面分析
    print("开始进行训练结果分析...")
    try:
        # 使用analyze_results模块进行完整分析
        data = load_results(config.SAVE_DIR)
        
        # 绘制训练指标图表
        metrics_path = analyzer.plot_training_metrics(data, config.SAVE_DIR)
        print(f"训练指标图表已保存至: {metrics_path}")
        
        # 绘制动作热力图
        heatmap_path = analyzer.plot_action_heatmap(model, config.SAVE_DIR)
        if heatmap_path:
            print(f"策略热力图已保存至: {heatmap_path}")
        
        # 生成演示视频
        def create_test_env():
            env = CustomPongEnv()
            return env
            
        video_dir = analyzer.generate_video_demo(model, config.SAVE_DIR, create_test_env)
        if video_dir:
            print(f"演示视频已保存至: {video_dir}")
        
        # 导出统计数据
        log_path, stats_path = analyzer.export_statistics(data, config.SAVE_DIR)
        print(f"统计数据已导出至: {stats_path}")
        
    except Exception as e:
        print(f"分析过程发生错误: {str(e)}")
    
    print(f"所有任务完成，结果已保存至: {config.SAVE_DIR}")
    env.close()

if __name__ == "__main__":
    main()