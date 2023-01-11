#= POLYNOMIAL LINEAR REGRESSION =#

## Import Librairies
using CSV, DataFrames
using Plots
using MLJ

## Read Data From File
df = CSV.read("../../Datasets/Position_Salaries.csv", DataFrame)
schema(df)

x = select(df, :Level)
n = 4
X = zeros(size(x)[1], n) # length(x.Level)
y = df.Salary

## Partition Of Data
train, test = partition(eachindex(y), 0.8, shuffle=true, rng=123)
Xtrain, Xtest = X[train, :], X[test, :]
ytrain, ytest = y[train], y[test]

## Load Linear Regressor
LR = @load LinearRegressor pkg=MLJLinearModels
lr = LR()

## Features Scaling
sc = Standardizer()
for i in 1:n
    mach_sc = machine(sc, x.Level[train].^i) |> fit!
    Xtrain[:, i] = MLJ.transform(mach_sc, x.Level[train].^i)
    Xtest[:, i] = MLJ.transform(mach_sc, x.Level[test].^i)
end

## Train & Fit The Regressor
mach = machine(lr, table(Xtrain), ytrain) |> fit!
params = fitted_params(mach)

## Predict & Measure The Error
yhat = predict(mach, table(Xtest))
println("Error is $(sum(( yhat.- ytest ).^2))")

## Plot & Compare
using LaTeXStrings
plot(x.Level[test], ytest, label=L"y(t)")
plot!(x.Level[test], yhat, label=L"\hat{y}(t)")
