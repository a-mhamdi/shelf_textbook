using CSV, DataFrames
using MLJ

df = CSV.read("../Datasets/Position_Salaries.csv", DataFrame)
describe(df)
