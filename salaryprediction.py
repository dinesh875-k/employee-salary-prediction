
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, r2_score

df = pd.read_csv('Updated_Employee_Salary_Dataset_Adjusted.csv')

df

df.info()

df.describe()

df.shape

df = df.drop(columns=['ID'])

df.head()

df.tail()

df.isnull().sum()        # Count missing values per column

#Visualize missing data patterns using heatmaps
import seaborn as sns
import matplotlib.pyplot as plt
sns.heatmap(df.isnull(), cbar=True, cmap='plasma')
plt.show()

label_encoder = LabelEncoder()
df['Gender'] = label_encoder.fit_transform(df['Gender'])  # Male=1, Female=0

X = df[['Experience_Years', 'Age', 'Gender']]
y = df['Salary']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

linear_model = LinearRegression()
linear_model.fit(X_train, y_train)
linear_preds = linear_model.predict(X_test)
linear_mae = mean_absolute_error(y_test, linear_preds)
linear_r2 = r2_score(y_test, linear_preds)

print("Linear Regression:")
print(f"  Coefficients: {linear_model.coef_}")
print(f"  Intercept: {linear_model.intercept_}")
print(f"  MAE: ₹{linear_mae:,.0f}")
print(f"  R² Score: {linear_r2:.2f}\n")

# 2. Random Forest Regressor
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_preds = rf_model.predict(X_test)
rf_mae = mean_absolute_error(y_test, rf_preds)
rf_r2 = r2_score(y_test, rf_preds)

print("Random Forest Regressor:")
print(f"  MAE: ₹{rf_mae:,.0f}")
print(f"  R² Score: {rf_r2:.2f}")

# Example: Predict salary for a new employee
def predict_salary(age, experience, gender_str):
    gender = label_encoder.transform([gender_str])[0]
    input_data = pd.DataFrame([[experience, age, gender]], columns=['Experience_Years', 'Age', 'Gender'])
    linear_salary = linear_model.predict(input_data)[0]
    rf_salary = rf_model.predict(input_data)[0]
    return {
        "Linear Regression Prediction": round(linear_salary, 2),
        "Random Forest Prediction": round(rf_salary, 2)
    }

# Example usage
print(predict_salary(age=30, experience=5, gender_str='Male'))

import pickle

with open('salary_model.pkl', 'wb') as file:
    pickle.dump(linear_model, file)




