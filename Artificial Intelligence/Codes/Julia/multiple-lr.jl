using CSV, DataFrames
using MLJ

df = CSV.read("../Datasets/50_Startups.csv", DataFrame)
describe(df)

