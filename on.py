import torch
import torch.onnx
import numpy as np

# 加载 TorchScript 模型
model = torch.jit.load("carbon_emission_model.pt")
model.eval()

# 示例输入数据
test_data = {
    'Body Type': 'normal',
    'Sex': 'female',
    'Diet': 'omnivore',
    'How Often Shower': 'less frequently',
    'Heating Energy Source': 'electricity',
    'Transport': 'public',
    'Social Activity': 'sometimes',
    'Monthly Grocery Bill': 190.0,
    'Frequency of Traveling by Air': 'never',
    'Vehicle Monthly Distance Km': 15.0,
    'Waste Bag Size': 'medium',
    'Waste Bag Weekly Count': 3,
    'How Long TV PC Daily Hour': 8.0,
    'How Many New Clothes Monthly': 0,
    'How Long Internet Daily Hour': 12.0,
    'Energy efficiency': 'No',
    'Recycling_Glass': 0,
    'Recycling_Metal': 0,
    'Recycling_Paper': 1,
    'Recycling_Plastic': 0,
    'Cooking_With_Airfryer': 0,
    'Cooking_With_Grill': 0,
    'Cooking_With_Microwave': 0,
    'Cooking_With_Oven': 0,
    'Cooking_With_Stove': 0
}

# 将 test_data 转换为模型输入格式
# 假设模型的输入是一个包含所有特征的向量
input_data = [
    # 数值特征
    test_data['Monthly Grocery Bill'],
    test_data['Vehicle Monthly Distance Km'],
    test_data['Waste Bag Weekly Count'],
    test_data['How Long TV PC Daily Hour'],
    test_data['How Many New Clothes Monthly'],
    test_data['How Long Internet Daily Hour'],
    test_data['Recycling_Glass'],
    test_data['Recycling_Metal'],
    test_data['Recycling_Paper'],
    test_data['Recycling_Plastic'],
    test_data['Cooking_With_Airfryer'],
    test_data['Cooking_With_Grill'],
    test_data['Cooking_With_Microwave'],
    test_data['Cooking_With_Oven'],
    test_data['Cooking_With_Stove'],
    # 分类特征需要转换为数值（独热编码）
    # Body Type
    1 if test_data['Body Type'] == 'overweight' else 0,
    1 if test_data['Body Type'] == 'obese' else 0,
    1 if test_data['Body Type'] == 'underweight' else 0,
    1 if test_data['Body Type'] == 'normal' else 0,
    # Sex
    1 if test_data['Sex'] == 'female' else 0,
    1 if test_data['Sex'] == 'male' else 0,
    # Diet
    1 if test_data['Diet'] == 'pescatarian' else 0,
    1 if test_data['Diet'] == 'vegetarian' else 0,
    1 if test_data['Diet'] == 'omnivore' else 0,
    1 if test_data['Diet'] == 'vegan' else 0,
    # How Often Shower
    1 if test_data['How Often Shower'] == 'daily' else 0,
    1 if test_data['How Often Shower'] == 'less frequently' else 0,
    1 if test_data['How Often Shower'] == 'more frequently' else 0,
    1 if test_data['How Often Shower'] == 'twice a day' else 0,
    # Heating Energy Source
    1 if test_data['Heating Energy Source'] == 'coal' else 0,
    1 if test_data['Heating Energy Source'] == 'natural gas' else 0,
    1 if test_data['Heating Energy Source'] == 'wood' else 0,
    1 if test_data['Heating Energy Source'] == 'electricity' else 0,
    # Transport
    1 if test_data['Transport'] == 'public' else 0,
    1 if test_data['Transport'] == 'walk/bicycle' else 0,
    1 if test_data['Transport'] == 'private (petrol)' else 0,
    1 if test_data['Transport'] == 'private (diesel)' else 0,
    1 if test_data['Transport'] == 'private (hybrid)' else 0,
    1 if test_data['Transport'] == 'private (lpg)' else 0,
    1 if test_data['Transport'] == 'private (electric)' else 0,
    # Social Activity
    1 if test_data['Social Activity'] == 'often' else 0,
    1 if test_data['Social Activity'] == 'never' else 0,
    1 if test_data['Social Activity'] == 'sometimes' else 0,
    # Frequency of Traveling by Air
    1 if test_data['Frequency of Traveling by Air'] == 'frequently' else 0,
    1 if test_data['Frequency of Traveling by Air'] == 'rarely' else 0,
    1 if test_data['Frequency of Traveling by Air'] == 'never' else 0,
    1 if test_data['Frequency of Traveling by Air'] == 'very frequently' else 0,
    # Waste Bag Size
    1 if test_data['Waste Bag Size'] == 'large' else 0,
    1 if test_data['Waste Bag Size'] == 'extra large' else 0,
    1 if test_data['Waste Bag Size'] == 'small' else 0,
    1 if test_data['Waste Bag Size'] == 'medium' else 0,
    # Energy efficiency
    1 if test_data['Energy efficiency'] == 'No' else 0,
    1 if test_data['Energy efficiency'] == 'Sometimes' else 0,
    1 if test_data['Energy efficiency'] == 'Yes' else 0
]

# 检查输入数据的长度
print("Input data length:", len(input_data))  # 应该输出 54

# 转换为 PyTorch 张量
input_tensor = torch.tensor([input_data], dtype=torch.float32)

# 导出为 ONNX 格式
torch.onnx.export(
    model,
    input_tensor,
    "carbon_emission_model.onnx",
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    opset_version=11
)

print("Model exported to carbon_emission_model.onnx")