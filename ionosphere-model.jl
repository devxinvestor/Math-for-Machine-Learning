using CSV
using DataFrames
using ECOS
using Convex
using Statistics

data = CSV.read("ionosphere.data", DataFrame)

X = Matrix(data[:, 1:end-1])
y = data[:, end]

y = map(yi -> yi == "b" ? 0 : 1, y)

X_train = X[1:300, :]
y_train = y[1:300]
X_test = X[301:end, :]
y_test = y[301:end]

n_features = size(X, 2)
w = Variable(n_features)
v = Variable()

function least_squares_loss(s)
    return square(s - 1)
end

function logistic_loss(s)
    return log(1 + exp(-s))
end

function hinge_loss(s)
    return max(0, 1 - s)
end

function train_model(X, y, loss_function)
    problem = minimize(sum(loss_function(y[i] * (X[i, :]' * w + v)) for i in 1:size(X, 1)))
    solve!(problem, () -> ECOS.Optimizer())
    return w.value, v.value
end

function predict(X, w, v)
    return X * w .+ v
end

function calculate_accuracy(y_true, y_pred)
    return mean((y_pred .> 0.5) .== (y_true .== 1))
end

w_value, v_value = train_model(X_train, y_train, least_squares_loss)
y_pred_ls = predict(X_test, w_value, v_value)

w_value, v_value = train_model(X_train, y_train, logistic_loss)
y_pred_logistic = predict(X_test, w_value, v_value)

w_value, v_value = train_model(X_train, y_train, hinge_loss)
y_pred_hinge = predict(X_test, w_value, v_value)

ls_accuracy = calculate_accuracy(y_test, y_pred_ls)
logistic_accuracy = calculate_accuracy(y_test, y_pred_logistic)
hinge_accuracy = calculate_accuracy(y_test, y_pred_hinge)

println("Lease Squares Loss Accuracy: ", ls_accuracy)
println("Logistic Loss Accuracy: ", ls_accuracy)
println("Hinge Loss Accuracy: ", hinge_accuracy)

ls_y_pred = map(yi -> yi > 0.5 ? "g" : "b", y_pred_ls)
logistic_y_pred = map(yi -> yi > 0.5 ? "g" : "b", y_pred_logistic)
hinge_y_pred = map(yi -> yi > 0.5 ? "g" : "b", y_pred_hinge)