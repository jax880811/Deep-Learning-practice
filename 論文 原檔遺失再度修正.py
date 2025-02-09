import numpy as np

# 環境設定
GRID_SIZE = 3  #設定網格大小為 3×3，總共有 9 個狀態。
ACTIONS = ['up', 'down', 'left', 'right']
ALPHA = 0.1  # 學習率 用來控制 Q 值更新的幅度：Q(s,a)←Q(s,a)+α⋅[TD誤差]
GAMMA = 0.9  #折扣因子 (γ)，用來平衡 當前獎勵 和 未來回報。
EPISODES = 500 #設定訓練 1000 次，每次從s1走到s9

# 初始化 Q 表
q_table = np.zeros((GRID_SIZE, GRID_SIZE, len(ACTIONS))) #這行建立一個三維陣列來存儲 Q 值：
'''
第一維：x 座標 (row)
第二維：y 座標 (column)
第三維：四個動作的 Q 值
'''
# 獎勵函數
def reward_function(state, action):
    # 到達終點的獎勵最高
    if state == (2, 2):  # 到達終點 就給10獎勵點
        return 10
    # 碰到邊界或向左向上移動有懲罰
    if action == 0 and state[0] > 0:  # 向上
        return -1
    elif action == 2 and state[1] > 0:  # 向左
        return -1
    # 其他情況給予較小的負獎勵，鼓勵盡快到達終點
    else:
        return -0.1

# 狀態轉移函數 根據當前狀態和動作，計算下一個狀態 (next_state)。也有避免出界的功能
def transition(state, action):
    x, y = state
    if action == 0:  # up
        next_state = (max(0, x - 1), y)
    elif action == 1:  # down
        next_state = (min(x + 1, GRID_SIZE - 1), y)
    elif action == 2:  # left
        next_state = (x, max(0, y - 1))
    elif action == 3:  # right
        next_state = (x, min(y + 1, GRID_SIZE - 1))
    reward = reward_function(next_state, action)
    return next_state, reward

# 選擇動作
def choose_action(state, epsilon):
    if np.random.uniform(0, 1) < epsilon:
        # 探索
        return np.random.choice(len(ACTIONS))
    else:
        # 利用
        return np.argmax(q_table[state[0], state[1]])
'''
ε-greedy 演算法：
以 ε 的機率隨機選擇動作 (探索)。
以 1-ε 的機率選擇當前最好的 Q 值 (利用)。
'''
# 更新 Q 表

def update_q_table(state, action, reward, next_state, alpha, gamma):
    q_predict = q_table[state[0], state[1], action]
    q_target = reward + gamma * np.max(q_table[next_state[0], next_state[1]])
    q_table[state[0], state[1], action] = q_predict + alpha * (q_target - q_predict)

# 訓練過程
def train(episodes, alpha, gamma):
    for episode in range(episodes):
        state = (0, 0)  # 從起始狀態開始
        done = False
        while not done:
            action = choose_action(state, epsilon=0.1)  # 探索與利用
            next_state, reward = transition(state, action)  # 執行動作，獲取新狀態和獎勵
            update_q_table(state, action, reward, next_state, alpha, gamma)
            state = next_state
            if state == (2, 2):  # 到達終點
                done = True

def calculate_probabilities(state):
    q_values = q_table[state[0], state[1]]
    total_q_value = np.sum(q_values)
    
    # 計算向上和向右的Q之和
    up_right_prob = (q_values[0] + q_values[3]) / total_q_value
    
    # 計算向下和向左的Q之和
    down_left_prob = (q_values[1] + q_values[2]) / total_q_value
    
    return up_right_prob, down_left_prob

def find_optimal_path():
    state = (0, 0)
    path = [state]
    while state != (2, 2):
        action = np.argmax(q_table[state[0], state[1], :])  # 修正此行
        next_state, _ = transition(state, action)
        state = next_state
        path.append(state)

    # 將數字狀態轉換為字母表示
    state_names = ['S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8', 'S9']
    path_str = [state_names[i * GRID_SIZE + j] for i, j in path]
    print("最優路徑：", ' -> '.join(path_str))
    prob_to_s2_or_s3 = calculate_probabilities((0, 0))
    print("從S1到S2以及從S1到S3的機率為何:", prob_to_s2_or_s3)
# 訓練模型
train(EPISODES, ALPHA, GAMMA)
# 找到最優路徑
find_optimal_path()