import os
from flask import Flask, request, jsonify, render_template
import input as it  # 假设这是你计算碳足迹的模块
from flask_cors import CORS
import joblib  # 用来加载 .pkl 模型

app = Flask(__name__)
CORS(app)

app.config['STATIC_FOLDER'] = os.path.join(os.getcwd(), 'static')

# 加载 .pkl 模型
model = joblib.load('carbon_emission_model.pkl')

@app.route('/')
def index():
    # 返回静态 HTML 页面
    return render_template('index.html')

@app.route('/calculate', methods=['POST'])
def calculate():
    # 从请求中获取数据
    data = request.get_json()

    # 构造用户数据字典
    user_data = {
        'Body Type': data.get('bodyType', 'normal'),
        'Sex': data.get('sex', 'female'),
        'Diet': data.get('diet', 'vegan'),
        'How Often Shower': data.get('howOftenShower', 'daily'),
        'Heating Energy Source': data.get('heatingEnergySource', 'coal'),
        'Transport': data.get('transport', 'public'),
        'Monthly Grocery Bill': float(data.get('monthlyGroceryBill', 0)),
        'Frequency of Traveling by Air': data.get('frequencyOfTravelingByAir', 'never'),
        'Vehicle Monthly Distance Km': float(data.get('vehicleMonthlyDistance', 0)),
        'Social Activity': data.get('socialActivity', 'sometimes'),
        'Waste Bag Si+S1+L1:T2': data.get('wasteBag', 'medium'),
        'Waste Bag Weekly Count': data.get('wasteBagWeeklyCount', 3),
        'How Long TV PC Daily Hour': float(data.get('howLongTVPCDailyHour', 3)),
        'How Many New Clothes Monthly': data.get('howManyNewClothesMonthly', 1),
        'How Long Internet Daily Hour': float(data.get('howLongInternetDailyHour', 3.0)),
        'Energy efficiency': data.get('energyEfficiency', 'Yes'),
        'Recycling_Glass': 1 if data.get('recyclingGlass') else 0,
        'Recycling_Metal': 1 if data.get('recyclingMetal') else 0,
        'Recycling_Paper': 1 if data.get('recyclingPaper') else 0,
        'Recycling_Plastic': 1 if data.get('recyclingPlastic') else 0,
        'Cooking_With_Airfryer': 1 if data.get('cookingWithAirfryer') else 0,
        'Cooking_With_Grill': 1 if data.get('cookingWithGrill') else 0,
        'Cooking_With_Microwave': 1 if data.get('cookingWithMicrowave') else 0,
        'Cooking_With_Oven': 1 if data.get('cookingWithOven') else 0,
        'Cooking_With_Stove': 1 if data.get('cookingWithStove') else 0
    }

    # 调用计算函数，并将数据传入模型进行预测
    result = it.calculate(user_data)

    # 返回计算结果
    return jsonify({"total_emission": result})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
