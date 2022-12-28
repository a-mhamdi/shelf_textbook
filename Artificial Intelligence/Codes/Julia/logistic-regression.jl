using DataFrames, CSV
using MLJ

df = CSV.read("../Datasets/***")
describe(df)
schema(df)

