from flask import Flask, request, jsonify
import numpy as np
import onnxruntime as ort

app = Flask(__name__)

# 加载 ONNX 模型
model_path = "carbon_emission_model.onnx"
session = ort.InferenceSession(model_path)

@app.route('/calculate', methods=['POST'])
def calculate():
    # 获取前端传递的数据
    data = request.json

    # 根据货币单位转换食品消费金额
    if data['currencyType'] == 'RMB':
        # 假设 1 USD = 7 RMB
        data['monthlyGroceryBill'] = data['monthlyGroceryBill'] / 7

    # 将数据转换为模型输入格式
    input_data = prepare_input_data(data)

    # 运行模型
    input_tensor = np.array(input_data, dtype=np.float32).reshape(1, -1)
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    result = session.run([output_name], {input_name: input_tensor})

    # 返回结果
    return jsonify({"totalEmission": float(result[0][0])})

def prepare_input_data(data):
    # 将前端数据转换为模型输入格式
    input_data = [
        data['vehicleMonthlyDistance'],
        data['howManyNewClothesMonthly'],
        data['wasteBagWeeklyCount'],
        data['howLongTVPCDailyHour'],
        data['howLongInternetDailyHour'],
        data['monthlyGroceryBill'],
        # 分类特征的独热编码
        *one_hot_encode(data['bodyType'], ['overweight', 'obese', 'underweight', 'normal']),
        *one_hot_encode(data['sex'], ['female', 'male']),
        *one_hot_encode(data['diet'], ['pescatarian', 'vegetarian', 'omnivore', 'vegan']),
        *one_hot_encode(data['showerFrequency'], ['daily', 'less frequently', 'more frequently', 'twice a day']),
        *one_hot_encode(data['heatingEnergySource'], ['coal', 'natural gas', 'wood', 'electricity']),
        *one_hot_encode(data['transport'], ['public', 'walk/bicycle', 'private (petrol)', 'private (diesel)', 'private (hybrid)', 'private (lpg)', 'private (electric)']),
        *one_hot_encode(data['socialActivity'], ['often', 'never', 'sometimes']),
        *one_hot_encode(data['airTravelFrequency'], ['frequently', 'rarely', 'never', 'very frequently']),
        *one_hot_encode(data['wasteBagSize'], ['large', 'extra large', 'small', 'medium']),
        *one_hot_encode(data['energyEfficiency'], ['No', 'Sometimes', 'Yes']),

        data['recyclingGlass'],
        data['recyclingMetal'],
        data['recyclingPaper'],
        data['recyclingPlastic'],
        data['cookingWithAirfryer'],
        data['cookingWithGrill'],
        data['cookingWithMicrowave'],
        data['cookingWithOven'],
        data['cookingWithStove']
    ]
    return input_data

def one_hot_encode(value, options):
    # 独热编码
    return [1 if option == value else 0 for option in options]

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
