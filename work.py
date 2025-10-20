import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
from sklearn.linear_model import Lasso
import matplotlib.pyplot as plt

# Load and remove duplicates
total_data = pd.read_csv("demographic_health_data.csv")
# print(total_data.head())
total_data = total_data.drop_duplicates().reset_index(drop = True)
# print(total_data.head())

# Return numeric columns
numeric_columns = (total_data.select_dtypes(include="number").columns.drop("Heart disease_number", errors="ignore"))

#  Standarize
scaler = StandardScaler()
norm_features = scaler.fit_transform(total_data[numeric_columns])

# Create dataframe and add target
scaled_data = pd.DataFrame(norm_features, index = total_data.index, columns = numeric_columns)
scaled_data["Heart disease_number"] = total_data["Heart disease_number"]

# Split features
x = scaled_data.drop(columns=["Heart disease_number"])
y = scaled_data["Heart disease_number"]
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.2, random_state = 1)

# Select 30 features
select_best = SelectKBest(f_regression, k = 30)
select_best.fit(x_train, y_train)

# Returns columns with strongest correlation (TRUE/FALSE)
selected_columns = select_best.get_support()


# Trims the training and test data
x_train_selected = pd.DataFrame(select_best.transform(x_train), columns = x_train.columns.values[selected_columns])
x_test_selected = pd.DataFrame(select_best.transform(x_test), columns = x_test.columns.values[selected_columns])


#  Testing linear regression model

model = LinearRegression()
model.fit(x_train_selected, y_train)

predict_y = model.predict(x_test_selected)
# print(predict_y)


# print(f"MSE: {mean_squared_error(y_test, predict_y)}")
print(f"Linear Regression R2 Score: {r2_score(y_test, predict_y)}")
# print(f"Average Difference compared to actual value {round(np.sqrt(10028986.47611465),2)}")


# Testing lasso model
lasso_model = Lasso(alpha = 1, max_iter=1000)

lasso_model.fit(x_train_selected, y_train)

print(f"Lasso Model R2 score:: {lasso_model.score(x_test_selected, y_test)}")



# Overfitting? Data leaks?

# print(len(x_train.columns))
# print(scaled_data.head())