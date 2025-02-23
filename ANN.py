import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import MultiLabelBinarizer
import ast

data = pd.read_csv('dataset/emission.csv')

for index, row in data.iterrows():
    transport = row['Transport']
    vehicle_type = row['Vehicle Type']

    if transport == 'private' and pd.isna(vehicle_type):
        raise ValueError(f"Row {index}: Vehicle Type must be provided when Transport is 'private'.")
    elif transport in ['public', 'walk/bicycle'] and not pd.isna(vehicle_type):
        raise ValueError(f"Row {index}: Vehicle Type must be NaN when Transport is '{transport}'.")

data['Transport']=data.apply(
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

X = data.drop(columns=['CarbonEmission'])
y = data['CarbonEmission']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
categorical_features = X.select_dtypes(include=['object']).columns
numeric_features = numeric_features.union(recycling_columns + cooking_columns)
"""print(numeric_features)
print("_________________")
print(categorical_features)"""

if __name__ == '__main__':
    for name in X.columns:
        print(f'{name} : {X[name].unique()}')

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(), categorical_features)
        ])

    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', MLPRegressor(hidden_layer_sizes=(64,64,32,32,16), max_iter=10000, random_state=42, learning_rate_init=0.00001))
    ])

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f'Mean Squared Error: {mse}')
    print(f'Root Mean Squared Error: {rmse}')
    print(f'Mean Absolute Error: {mae}')
    print(f'R² Score: {r2}')

    #joblib.dump(model, 'carbon_emission_model.pkl') #梯度消失/梯度爆炸