import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import MultiLabelBinarizer
import ast
import joblib

# 加载数据
data = pd.read_csv('dataset/emission.csv')

# 数据预处理
for index, row in data.iterrows():
    transport = row['Transport']
    vehicle_type = row['Vehicle Type']

    if transport == 'private' and pd.isna(vehicle_type):
        raise ValueError(f"Row {index}: Vehicle Type must be provided when Transport is 'private'.")
    elif transport in ['public', 'walk/bicycle'] and not pd.isna(vehicle_type):
        raise ValueError(f"Row {index}: Vehicle Type must be NaN when Transport is '{transport}'.")

data['Transport'] = data.apply(
    lambda row: f"{row['Transport']} ({row['Vehicle Type']})" if pd.notna(row['Vehicle Type']) else row['Transport'],
    axis=1
)

data.drop(columns=['Vehicle Type'], inplace=True)

data['Recycling'] = data['Recycling'].apply(lambda x: ast.literal_eval(x) if pd.notna(x) else [])
data['Cooking_With'] = data['Cooking_With'].apply(lambda x: ast.literal_eval(x) if pd.notna(x) else [])

mlb_recycling = MultiLabelBinarizer()
mlb_cooking = MultiLabelBinarizer()

recycling_encoded = mlb_recycling.fit_transform(data['Recycling'])
recycling_columns = [f'Recycling_{item}' for item in mlb_recycling.classes_]
recycling_df = pd.DataFrame(recycling_encoded, columns=recycling_columns)

cooking_encoded = mlb_cooking.fit_transform(data['Cooking_With'])
cooking_columns = [f'Cooking_With_{item}' for item in mlb_cooking.classes_]
cooking_df = pd.DataFrame(cooking_encoded, columns=cooking_columns)

data = pd.concat([data, recycling_df, cooking_df], axis=1)
data.drop(columns=['Recycling', 'Cooking_With'], inplace=True)

# 特征和目标变量
X = data.drop(columns=['CarbonEmission'])
y = data['CarbonEmission']

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 数值特征和分类特征
numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
categorical_features = X.select_dtypes(include=['object']).columns
numeric_features = numeric_features.union(recycling_columns + cooking_columns)

# 数据预处理
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(), categorical_features)
    ])

# 对训练集和测试集进行预处理
X_train = preprocessor.fit_transform(X_train)
X_test = preprocessor.transform(X_test)

# 转换为 PyTorch 张量
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)


# 定义 PyTorch 模型
class EmissionModel(nn.Module):
    def __init__(self, input_size):
        super(EmissionModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# 初始化模型
input_size = X_train.shape[1]
model = EmissionModel(input_size)

# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 创建 DataLoader
train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# 训练模型
num_epochs = 100
for epoch in range(num_epochs):
    for inputs, targets in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# 测试模型
model.eval()
with torch.no_grad():
    y_pred = model(X_test)
    y_pred = y_pred.numpy()

# 计算评估指标
mse = mean_squared_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared=False)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse}')
print(f'Root Mean Squared Error: {rmse}')
print(f'Mean Absolute Error: {mae}')
print(f'R² Score: {r2}')

# 导出模型为 TorchScript 格式
model_scripted = torch.jit.script(model)  # 将模型转换为 TorchScript
model_scripted.save("carbon_emission_model.pt")  # 保存模型
print("Model exported to carbon_emission_model.pt")

# 保存预处理对象
joblib.dump(preprocessor, 'preprocessor.pkl')
print("Preprocessor saved to preprocessor.pkl")
