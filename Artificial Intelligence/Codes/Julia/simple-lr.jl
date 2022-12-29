using CSV, DataFrames
using MLJ
## Read data using .csv file. Convert it to DataFrame object
df = CSV.read("../Datasets/Salary_Data.csv", DataFrame)
## Unpacking Features and Target
X = select(df, :YearsExperience) # |> Tables.matrix |> table
y = df.Salary # |> Tables.matrix |> table
## Preparing for the split
# train, test = partition(eachindex(y), 0.8, shuffle=true, rng=123)
## Load and instantiate the linear regressor object
LR = @load LinearRegressor pkg=MLJLinearModels
lr = LR()
mach = machine(lr, X, y)
fit!(mach)
yhat = predict(mach, X)
fitted_params(mach)
println("Error is $(sum( (yhat .- y).^2 ))")
