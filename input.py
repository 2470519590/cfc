import joblib
import pandas as pd


def calculate(input_data):
    model = joblib.load('carbon_emission_model.pkl')
    input_df = pd.DataFrame([input_data])
    predicted_emission = model.predict(input_df)
    return int(predicted_emission[0])


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
        'Waste Bag Si+S1+L1:T2': 'medium',
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


if __name__ == '__main__':
    print(calculate(test_data))