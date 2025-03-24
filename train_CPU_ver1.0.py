import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import os
import logging
from dotenv import load_dotenv
from typing import Any, Dict
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.results_plotter import load_results
from custom_pong_env import CustomPongEnv  # 确保文件名匹配
from generate_reward import generate_reward_function, validate_reward  # 确保文件名匹配

# 强制使用无交互式后端
plt.switch_backend('Agg')

# 指定 .env 文件路径（假设在当前目录）
script_dir = os.path.dirname(os.path.abspath(__file__))
env_path = os.path.join(script_dir, ".env")  # 确保 .env 与脚本同级

load_dotenv(env_path)
# load_dotenv(os.path.join(os.getcwd(), ".env"))
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("train.log"), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

class Config:
    TOTAL_TIMESTEPS: int = 100_000
    SAVE_FREQ: int = 20_000
    # 使用用户目录，避免权限问题
    SAVE_DIR: str = os.path.expanduser("~\pong_saved_models")
    POLICY: str = "MlpPolicy"

config = Config()

class SaveModelCallback(BaseCallback):
    def __init__(self, save_freq: int, save_path: str, verbose: int = 0):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path

    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            save_path = os.path.join(self.save_path, f"model_{self.n_calls}")
            try:
                self.model.save(save_path)
            except PermissionError as e:
                logger.error(f"保存模型失败：{e}")
                return False
        return True

class RewardWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env, reward_func: Any):
        super().__init__(env)
        self.reward_func = reward_func

    def step(self, action: int) -> tuple:
        obs, _, terminated, truncated, info = self.env.step(action)
        reward = self.reward_func(obs)
        return obs, reward, terminated, truncated, info

def main():
    deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")
    if not deepseek_api_key:
        logger.critical("未找到DEEPSEEK_API_KEY环境变量")
        raise ValueError("API密钥未设置，请在.env文件中设置DEEPSEEK_API_KEY")

    reward_func = None
    for retry in range(3):
        try:
            code = generate_reward_function(deepseek_api_key)
            #logger.info(f"第{retry+1}次生成代码：\n{code}")
            reward_func = validate_reward(code)
            if reward_func:
                break
        except Exception as e:
            logger.error(f"生成失败：{str(e)}", exc_info=True)
    if reward_func is None:
        logger.critical("无法生成有效的奖励函数")
        raise RuntimeError("无法生成有效的奖励函数")

    # 创建并包装环境
    env = CustomPongEnv()
    env = RewardWrapper(env, reward_func)
    from datetime import datetime
    log_name = os.path.expanduser(f"~\pong_training_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    try:
        env = Monitor(env, filename=log_name)
    except PermissionError as e:
        logger.error(f"创建日志文件失败：{e}")
        raise

    # 训练模型
    model = PPO(config.POLICY, env, verbose=1)
    os.makedirs(config.SAVE_DIR, exist_ok=True)
    callback = SaveModelCallback(config.SAVE_FREQ, config.SAVE_DIR)
    model.learn(total_timesteps=config.TOTAL_TIMESTEPS, callback=callback)
    try:
        model.save(os.path.join(config.SAVE_DIR, "final_model"))
    except PermissionError as e:
        logger.error(f"保存最终模型失败：{e}")

    # 绘制训练曲线
    data = load_results(log_name)
    plt.figure(figsize=(10, 6))
    plt.plot(data["r"], label="LLM Reward")
    plt.xlabel("Training Steps")
    plt.ylabel("Episode Reward")
    plt.title("Training Progress")
    plt.legend()
    plt.savefig(os.path.expanduser("~\pong_training_curve.png"))
    plt.close()  # 关闭图表

    env.close()  # 确保环境关闭

if __name__ == "__main__":
    main()