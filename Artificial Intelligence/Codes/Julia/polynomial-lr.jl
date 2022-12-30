#= POLYNOMIAL LINEAR REGRESSION =#

## Import Librairies
using CSV, DataFrames
using MLJ

## Read Data From File
df = CSV.read("../Datasets/Position_Salaries.csv", DataFrame)
schema(df)

X = df.Level
y = df.Salary

## Partition Of Data
train, test = partition(eachindex(y), 0.8, shuffle=true, rng=123)
Xtrain, Xtest = X[:, train], X[:, test]
ytrain, ytest = y[train], y[test]

## Load Linear Regressor
LR = @load LinearRegressor pkg=MLJLinearModels
lr = LR()

