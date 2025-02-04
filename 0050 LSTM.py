import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
'''
老實說不想要一直修改模型參數
不然要就把一開始的LSTM層的神經元從50砍到只剩下10好了:)
跑的神經元很多就真的很花時間
真希望有錢能夠換下CPU，看能不能跑快一點
'''

df=pd.read_csv('Taiwan ETF (0050.TW).csv', encoding='cp1252',header=0)

df['Date'] = pd.to_datetime(df['date'])
df = df[(df['Date'] >= '2009-01-05') & (df['Date'] <= '2024-09-27')]

df_features = df[['adj close', 'close', 'high', 'low', 'open']]

scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(df_features) #資料標準化

'''
LSTM 是一種特殊的 RNN（Recurrent Neural Network，循環神經網絡），主要用於處理和預測基於時間序列的數據。
它解決了標準 RNN 的 梯度消失 和 梯度爆炸 問題，使模型能有效學習長期依賴關係。
LSTM 的設計在於通過「記憶單元」來選擇性地保留或忘記時間序列中的信息。

LSTM 的主要組件包括：

記憶單元（Cell State）：

負責存儲長期記憶。
是 LSTM 的核心，通過「門控機制」來更新和維護信息。
門控機制（Gates）：

遺忘門（Forget Gate）：選擇需要遺忘的過去信息。
輸入門（Input Gate）：決定當前時刻的輸入信息應該添加到記憶單元中。
輸出門（Output Gate）：決定記憶單元中哪部分信息用於當前輸出。
LSTM 的主要參數
Input Shape：

time steps: 時間序列的長度，例如過去 30 天的數據。
features: 每個時間點的特徵數量，例如開盤價、收盤價等。
Number of Units：

LSTM 的神經元數量，表示該層的輸出維度。選擇較大的值可以學習更複雜的模式，但會增加計算成本。
Activation Function：

一般使用 tanh（隱藏層）和 sigmoid（門控機制），確保輸出值在特定範圍內。
Dropout：

防止過擬合的正則化技術，隨機關閉一定比例的神經元。
Recurrent Dropout：

對時間序列內的循環連接使用 dropout，進一步防止過擬合。
Batch Size：

一次訓練所處理的樣本數量，影響訓練效率和模型穩定性。
Optimizer：

使用例如 Adam 或 RMSprop 等優化器來更新權重。
Loss Function：

常用均方誤差（MSE）或平均絕對誤差（MAE）作為回歸任務的損失函數。
'''


#建立訓練數據集
X_train = []
y_train = []

#使用100天的窗口建立特徵和目標
for i in range(100, len(scaled_data)):
    X_train.append(scaled_data[i-100:i])#過去100天的數據
    y_train.append(scaled_data[i, 0])#第100天的 adj close 作為目標

X_train, y_train = np.array(X_train), np.array(y_train)

#檢查數據形狀
print("X_train shape:", X_train.shape)  #(樣本數, 100, 5)
print("y_train shape:", y_train.shape)


model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
    #X_train.shape[1] 和 X_train.shape[2] 是用來獲取數據維度（shape）中的具體數值。這些指令分別代表數據張量的第 2 維度和第 3 維度的大小
    #第2維度代表著100日的數據，第三維度則是特徵數量
    
    Dropout(0.2),#隨機扔掉20%的神經元，防止過擬合
    #不過題外話，感覺不需要扔掉欸:)，這模型的準確率我有點疑惑
    LSTM(50, return_sequences=False),#最後一層了，就不返回每個時間的輸出了，將結果作為密集層(Dense)的輸入
    Dropout(0.2),
    Dense(25, activation='relu'),#第一層25個位元，使用relu當作激勵函數，避免梯度消失的問題
    Dense(1)  # 預測單一值，作為輸出
])
'''
    第一層 LSTM:
    units=50: 每層有 50 個記憶單元。
    return_sequences=True: 返回每個時間步的輸出，供下一層 LSTM 使用。
    input_shape=(60, 5): 每個樣本有 60 個時間步，每個時間步包含 5 個特徵
    '''

# 編譯模型
model.compile(optimizer='adam', loss='mean_squared_error')

# 訓練模型
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.3)
'''
optimizer='adam': 自適應學習率方法，收斂速度快。
loss='mean_squared_error': 均方誤差作為損失函數，適合回歸任務。
validation_split=0.3: 30% 的數據作為驗證集，監控模型性能。
'''

# 查看模型摘要
model.summary()
'''
模型結構概覽
該模型由以下部分組成：

LSTM 層（第一層和第二層）
Dropout 層（防止過擬合）
全連接層（Dense 層）（用於將特徵映射到輸出空間）
輸出層（最終輸出）
總參數數量為 98,105，其中可訓練參數是 32,701，非訓練參數是 0。

LSTM 層有四個門（遺忘門、輸入門、輸出門和細胞狀態更新），每個門都有權重矩陣和偏置：
權重參數 = (input_dim+output_dim)×output_dim×4
偏置參數 = output_dim×4
該層參數:[(input_dim+output_dim)×output_dim]×4+[output_dim×4]
第一層權重參數 = (1+50)×50×4+50×4=11,200
到了第二層 input_dim變成了50個 於是參數變成(50+50)×50×4+50×4=20,200

權重參數 = input_dim * output_dim
偏置參數 = output_dim
公式依然是權重參數＋偏置參數
在這裡參數=50×25+25=1,275


'''

# 測試數據集
test_data = scaled_data[-365:]  #使用最後1年的數據
test_data = np.expand_dims(test_data, axis=0)  #添加維度
'''
test_data 是一個 NumPy 數組，其形狀為 (1, 100, 5)，表示：
1 個樣本。
每個樣本有 100 個時間步。
每個時間步有 5 個特徵。
'''

# 預測2025年365天的 adj close
predicted_adj_close = []
for _ in range(365):
    prediction = model.predict(test_data)
    predicted_adj_close.append(prediction[0][0])

    # 更新測試數據
    new_entry = np.hstack((np.squeeze(prediction), test_data[0, -1, 1:]))
    '''
    test_data[0, -1, 1:] 是對三維數據（例如 NumPy 數組或類似結構）進行索引操作，其形狀一般為 (samples, time_steps, features)。對應的含義是：

    0：選擇第一個樣本（索引為 0 的樣本）。
    -1：選擇最後一個時間步（索引為 -1 表示倒數第一個元素）。
    1:：對最後一個時間步的特徵進行切片，選擇從索引 1 到結尾的所有特徵。
    '''
    test_data = np.vstack((test_data[0, 1:], new_entry))
    test_data = np.expand_dims(test_data, axis=0)

# 還原數據到原始比例
predicted_adj_close = scaler.inverse_transform(
    np.hstack((np.array(predicted_adj_close).reshape(-1, 1), np.zeros((365, 4))))
)[:, 0]
'''
預測的數據從 [0,1] 反標準化回原始比例。
np.hstack: 假設其他欄位為 0，以適應反標準化的形狀需求。
'''

plt.figure(figsize=(12, 6))
plt.plot(range(1, 366), predicted_adj_close, label='2025 predict', color='blue')
plt.xlabel('2025 date')
plt.ylabel('adj close value')
plt.title('Taiwan ETF (0050.TW) 2025')
plt.legend()
plt.show()

'''
plt不能打中文喔:(
'''

# 分割測試集（2024/1/1 - 2024/9/27）
test_2024 = df[(df['Date'] >= '2024-01-01') & (df['Date'] <= '2024-09-27')]
test_features = test_2024[['adj close', 'close', 'high', 'low', 'open']].values
test_scaled = scaler.transform(test_features)

X_test = []
y_test = []

for i in range(100, len(test_scaled)):
    X_test.append(test_scaled[i-100:i])
    y_test.append(test_scaled[i, 0])

X_test, y_test = np.array(X_test), np.array(y_test)

#預測
y_pred = model.predict(X_test)

#還原數據
y_test_inverse = scaler.inverse_transform(np.hstack((y_test.reshape(-1, 1), np.zeros((len(y_test), 4)))))[:, 0]
y_pred_inverse = scaler.inverse_transform(np.hstack((y_pred, np.zeros((len(y_pred), 4)))))[:, 0]

#評估
from sklearn.metrics import mean_squared_error
rmse = np.sqrt(mean_squared_error(y_test_inverse, y_pred_inverse))
print(f"RMSE 2024 預測數值 : {rmse}")

#繪圖
plt.figure(figsize=(12, 6))
plt.plot(y_test_inverse, label='real Adj Close (2024)')
plt.plot(y_pred_inverse, label='predict Adj Close (2024)', color='red')
plt.xlabel('date')
plt.ylabel('adj close value')
plt.title('2024 performance')
plt.legend()
plt.show()


