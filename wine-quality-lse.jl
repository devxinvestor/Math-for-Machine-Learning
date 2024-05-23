using LinearAlgebra
using CSV
using DataFrames

data = CSV.read("winequality-red.csv", DataFrame)

X = Matrix(data[:, 1:end-1])
y = data[:, end]

X_train = X[1:1400, :]
y_train = y[1:1400]
X_test = X[1401:end, :]
y_test = y[1401:end]

lambdas = [0, 1e-3, 1e-2, 1e-1]

function mse(predictions, targets)
    return mean((predictions - targets).^2)
end

function solve_least_squares(X, y, lambda)
    Φ = vcat(hcat(X, ones(size(X, 1))), sqrt(lambda) * Matrix(I, size(X, 2) + 1, size(X, 2) + 1))
    ψ = vcat(y, zeros(size(X, 2) + 1))
    w_hat = (Φ' * Φ) \ (Φ' * ψ)
    return w_hat
end

for lambda in lambdas
    w_hat = solve_least_squares(X_train, y_train, lambda)
    predictions = X_test * w_hat[1:end-1] .+ w_hat[end]
    println("Lambda: ", lambda, ", MSE: ", mse(predictions, y_test))
end
