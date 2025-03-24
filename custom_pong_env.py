import gymnasium as gym  
from gymnasium import spaces  
import numpy as np  

class CustomPongEnv(gym.Env):  
    def __init__(self):  
        super().__init__()  
        # 状态变量：[球x坐标, 球y坐标, 板y坐标]，范围[0,1]  
        self.observation_space = spaces.Box(low=0, high=1, shape=(3,), dtype=np.float32)  
        # 动作空间：0-不动，1-上移，2-下移  
        self.action_space = spaces.Discrete(3)  
        self.reset()  

    def reset(self, seed=None, options=None):  
        super().reset(seed=seed, options=options)  
        # 初始状态：球在左侧，随机初始位置  
        self.ball_x = 0.1  
        self.ball_y = np.random.uniform(0.2, 0.8)  
        self.paddle_y = 0.5  
        # 随机初始球速（x方向固定向右，y方向随机）  
        self.ball_vx = 0.02  
        self.ball_vy = np.random.uniform(-0.03, 0.03)  
        return self._get_obs(), {}  

    def step(self, action):  
        # 更新球位置  
        self.ball_x += self.ball_vx  
        self.ball_y += self.ball_vy  

        # 边界反弹（y轴）  
        if self.ball_y <= 0 or self.ball_y >= 1:  
            self.ball_vy *= -1  
        self.ball_y = np.clip(self.ball_y, 0, 1)  # 限制范围

        # 更新板位置  
        if action == 1:  
            self.paddle_y = max(0.0, self.paddle_y - 0.04)  
        elif action == 2:  
            self.paddle_y = min(1.0, self.paddle_y + 0.04)  
        self.paddle_y = np.clip(self.paddle_y, 0, 1)  # 限制范围

        # 判断击球：球到达右侧且板在球附近
        hit = (self.ball_x >= 0.95) and (abs(self.paddle_y - self.ball_y) < 0.07)
        
        # 如果击中，球反弹回来
        if hit:
            self.ball_vx *= -1
            # 根据击球位置略微改变球的垂直速度，增加随机性
            self.ball_vy += (self.paddle_y - self.ball_y) * 0.1
            reward = 1.0  # 击球奖励
        else:
            reward = 0.0
        
        # 回合结束条件：球超出右边界(未击中)或超出左边界
        done = (self.ball_x >= 0.95 and not hit) or self.ball_x <= 0.0
        
        info = {"hit": hit}  # 添加击球信息供调试
        return self._get_obs(), reward, done, False, info 

          

    def _get_obs(self):  
        return np.array([self.ball_x, self.ball_y, self.paddle_y], dtype=np.float32)  