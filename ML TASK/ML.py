from linear_regression import *
file = 'UASVictory\ML TASK\Housing.csv'
x_train, x_test, y_train, y_test = data_split(file)
w_init = np.zeros(x_train.shape[1])
b_init = 0
iterations = 10000
tmp_alpha = 0.0007

w_final, b_final, J_hist = gradient_descent(x_train, y_train)

features = ["area", "bedrooms", "bathrooms", "stories", "mainroad", "guestroom", "basement", "hotwaterheating", "airconditioning", "parking", "prefarea", "furnishingstatus"]
target = "price"
y_pred = prediction(x_train, w_final, b_final)
mse, mae, r2 = calc_metrics(y_train, y_pred)
cost = cost(x_train, y_train, w_final, b_final)


# print(f"\nFinal parameters:")
# print(f"w: {w_final}")


y_pred_train = prediction(x_train, w_final, b_final)
y_pred_test = prediction(x_test, w_final, b_final)

print("\nFinal metrics:")
print(f"MSE: {mse:.6f}")
print(f"MAE: {mae:.6f}")
print(f"Rsquare: {r2:.6f}")


plot_graphs(J_hist, y_train, y_pred_train, y_test, y_pred_test)

# print(predict())