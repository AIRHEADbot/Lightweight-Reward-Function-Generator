import importlib
import numpy as np  
import math
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import os

# 设置无交互式后端，与 train_GPU_1.1.py 保持一致
plt.switch_backend('Agg')

# DPO 模型定义
class DPOModel(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=32):  # 增加隐藏层维度以提高表达能力
        super(DPOModel, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, state):
        return self.network(state)

def train_dpo_model(model, preference_data, epochs=200, lr=0.005, save_dir="dpo_models"):
    """
    使用偏好数据训练 DPO 模型并保存权重
    """
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()
    losses = []

    os.makedirs(save_dir, exist_ok=True)

    for epoch in range(epochs):
        total_loss = 0
        for state1, state2, preference in preference_data:
            state1 = torch.FloatTensor(state1)
            state2 = torch.FloatTensor(state2)
            pref = torch.FloatTensor([preference])
            
            reward1 = model(state1)
            reward2 = model(state2)
            logits = reward1 - reward2
            loss = loss_fn(logits, pref)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / len(preference_data)
        losses.append(avg_loss)
        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
    
    # 保存模型权重
    model_path = os.path.join(save_dir, "dpo_model.pth")
    torch.save(model.state_dict(), model_path)
    print(f"DPO 模型已保存至: {model_path}")

    # 可视化训练损失
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, epochs + 1), losses, label="Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("DPO Model Training Loss")
    plt.legend()
    plt.savefig(os.path.join(save_dir, "dpo_training_loss.png"))
    plt.close()
    print(f"训练损失图已保存至: {save_dir}/dpo_training_loss.png")

    return model, model_path

def generate_reward_function(preference_data=None, save_dir="dpo_models"):
    """
    使用 DPO 方法生成奖励函数，并保存模型权重以供调用
    """
    try:
        if preference_data is None:
            preference_data = [
                (np.array([0.9, 0.5, 0.5]), np.array([0.9, 0.5, 0.8]), 1),  # 板对齐优于未对齐
                (np.array([0.1, 0.3, 0.8]), np.array([0.1, 0.3, 0.3]), 0),  # 未对齐优于远离
                (np.array([0.95, 0.6, 0.6]), np.array([0.95, 0.6, 0.2]), 1), # 右边界对齐更优
                (np.array([0.5, 0.4, 0.4]), np.array([0.5, 0.4, 0.7]), 1),  # 中间位置对齐优
            ]
        
        model = DPOModel()
        trained_model, model_path = train_dpo_model(model, preference_data, save_dir=save_dir)
        
        # 生成奖励函数代码，嵌入模型结构和权重加载
        code = f"""def calculate_reward(state):
    import torch
    import numpy as np
    # Create a model with the same structure as DPOModel
    class DPOModel(torch.nn.Module):
        def __init__(self, input_dim=3, hidden_dim=32):
            super(DPOModel, self).__init__()
            self.network = torch.nn.Sequential(
                torch.nn.Linear(input_dim, hidden_dim),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden_dim, hidden_dim),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden_dim, 1)
            )
        
        def forward(self, state):
            return self.network(state)
    
    model = DPOModel()
    model.load_state_dict(torch.load("{model_path}"))
    model.eval()
    state_tensor = torch.FloatTensor(state)
    with torch.no_grad():
        reward = model(state_tensor).item()
    return float(reward)
"""
        return code
        
    except Exception as e:
        print(f"生成奖励函数失败: {str(e)}")
        return """def calculate_reward(state):
    ball_x = state[0]
    ball_y = state[1]
    paddle_y = state[2]
    vertical_distance = abs(paddle_y - ball_y)
    x_factor = ball_x ** 2
    reward = -vertical_distance * (1 + 5 * x_factor)
    return float(reward)"""

def extract_pure_code(text):  
    start_marker = "```python"  
    end_marker = "```"  
    start_idx = text.find(start_marker)  
    if start_idx == -1:  
        start_idx = text.find("def calculate_reward")  
        if start_idx == -1:  
            return None  
        code = text[start_idx:]  
    else:  
        start_idx += len(start_marker)  
        end_idx = text.find(end_marker, start_idx)  
        code = text[start_idx:end_idx]  
    return code.strip()

def validate_reward(code):  
    try:  
        spec = importlib.util.spec_from_loader("reward_module", loader=None)  
        module = importlib.util.module_from_spec(spec)
        module.__dict__["np"] = np
        module.__dict__["math"] = math  
        module.__dict__["torch"] = torch  
        exec(code, module.__dict__)  
        func = module.calculate_reward  
        test_state = np.array([0.5, 0.6, 0.5])  
        reward = func(test_state)  
        if not isinstance(reward, float):  
            raise ValueError("返回值必须是浮点数")  
        return func  
    except Exception as e:  
        print(f"验证失败：{e}\n生成的代码: \n{code}")  
        return None  

# 测试用例
'''def test_reward_function():
    print("测试 DPO 生成的奖励函数...")
    save_dir = "test_dpo_models"
    reward_code = generate_reward_function(save_dir=save_dir)
    reward_func = validate_reward(reward_code)
    
    if reward_func:
        test_states = [
            np.array([0.9, 0.5, 0.5]),  # 板与球对齐，靠近右边界
            np.array([0.1, 0.3, 0.8]),  # 板远离球，靠近左边界
            np.array([0.95, 0.6, 0.2]), # 板未对齐，靠近右边界
            np.array([0.5, 0.4, 0.4])   # 中间位置，板对齐
        ]
        print("\n测试结果：")
        for i, state in enumerate(test_states):
            reward = reward_func(state)
            print(f"状态 {i+1}: {state} -> 奖励: {reward:.4f}")
        
        # 可视化奖励分布
        ball_y = np.linspace(0, 1, 20)
        paddle_y = np.linspace(0, 1, 20)
        rewards = np.zeros((len(ball_y), len(paddle_y)))
        
        for i, by in enumerate(ball_y):
            for j, py in enumerate(paddle_y):
                state = np.array([0.9, by, py])  # 固定 ball_x = 0.9
                rewards[i, j] = reward_func(state)
        
        plt.figure(figsize=(10, 8))
        plt.imshow(rewards, cmap="viridis", origin="lower")
        plt.colorbar(label="Reward")
        plt.xticks(np.arange(0, 20, 5), np.round(paddle_y[::5], 2))
        plt.yticks(np.arange(0, 20, 5), np.round(ball_y[::5], 2))
        plt.xlabel("Paddle Y Position")
        plt.ylabel("Ball Y Position")
        plt.title("Reward Distribution (ball_x = 0.9)")
        plt.savefig(os.path.join(save_dir, "reward_heatmap.png"))
        plt.close()
        print(f"奖励分布热图已保存至: {save_dir}/reward_heatmap.png")
    else:
        print("奖励函数生成失败")

if __name__ == "__main__":
    test_reward_function()'''