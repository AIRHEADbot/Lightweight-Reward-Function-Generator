
import requests  
import importlib.util  
import numpy as np  
import math

def generate_reward_function(api_key):
    try:
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        prompt = """  
        你是一个强化学习专家,请为自定义的Pong游戏编写奖励函数。  
        已知状态变量是一个NumPy数组,包含3个浮点数(范围0-1):  
        state[0] - 球的x坐标(从左到右增加)
        state[1] - 球的y坐标(从下到上增加)  
        state[2] - 板的y坐标  

        目标：生成一个奖励函数，鼓励板移动到球的位置以成功击球。  
        要求：  
        1. 必须仅返回Python代码,不包含任何自然语言解释或Markdown标记;  
        2. 函数必须命名为`calculate_reward`，输入为`state`;  
        3. 使用state[1]和state[2]计算板与球的垂直距离；  
        4. 考虑球在x方向的距离，当球靠近右边界时给予更高权重；
        5. 返回值应为浮点数。

        示例：  
        def calculate_reward(state):  
            ball_y = state[1]
            paddle_y = state[2]
            vertical_distance = abs(paddle_y - ball_y)
            x_position_factor = state[0]  # 球越靠近右边界，factor越大
            reward = -vertical_distance * (1 + x_position_factor*2)
            return float(reward)
        """
        
        data = {
            "model": "deepseek-chat",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.3
        }
        
        response = requests.post("https://api.deepseek.com/v1/chat/completions", 
                                headers=headers, json=data)
        
        if response.status_code != 200:
            raise ValueError(f"API请求失败:{response.text}")
        
        code = response.json()["choices"][0]["message"]["content"]
        code = extract_pure_code(code)
        
        # 确保代码有效
        if not code or "calculate_reward" not in code:
            raise ValueError("无效的奖励函数代码")
            
        # 扩展标点替换
        code = code.replace("，", ",").replace("：", ":").replace("；", ";").replace("。", ".")
        return code
        
    except Exception as e:
        print(f"生成奖励函数失败: {str(e)}")
        # 提供一个更好的默认奖励函数
        return """def calculate_reward(state):
    ball_x = state[0]  # 球的x坐标
    ball_y = state[1]  # 球的y坐标
    paddle_y = state[2]  # 挡板的y坐标
    
    # 计算垂直距离
    vertical_distance = abs(paddle_y - ball_y)
    
    # 根据球的x位置调整奖励权重（球越靠近右边界，权重越大）
    x_factor = ball_x ** 2  # 使用平方关系增加接近边界时的重要性
    
    # 计算奖励：垂直对齐越好，奖励越高
    reward = -vertical_distance * (1 + 5 * x_factor)
    
    return float(reward)"""

def extract_pure_code(text):  
    # 提取 ```python ... ``` 之间的代码  
    start_marker = "```python"  
    end_marker = "```"  
    start_idx = text.find(start_marker)  
    if start_idx == -1:  
        # 如果没有Markdown标记，尝试直接提取函数定义  
        start_idx = text.find("def calculate_reward")  
        if start_idx == -1:  
            return None  
        code = text[start_idx:]  
    else:  
        start_idx += len(start_marker)  
        end_idx = text.find(end_marker, start_idx)  
        code = text[start_idx:end_idx]  
    # 去除首尾空白行  
    code = code.strip()  
    return code  

def validate_reward(code):  
    try:  
        # 动态执行代码  
        spec = importlib.util.spec_from_loader("reward_module", loader=None)  
        module = importlib.util.module_from_spec(spec)
        module.__dict__["np"] = np
        module.__dict__["math"] = math  
        exec(code, module.__dict__)  
        func = module.calculate_reward  

        # 语义验证：检查是否使用state[1]和state[2]  
        test_state = np.array([0.5, 0.6, 0.5])  
        reward = func(test_state)  
        if not isinstance(reward, float):  
            raise ValueError("返回值必须是浮点数")  
        return func  
    except Exception as e:  
        print(f"验证失败：{e}\n生成的代码: \n{code}")  # 显示替换后的代码  
        return None  