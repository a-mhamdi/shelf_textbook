#= Here is an example of how we might implement an `SVM`` in _Julia_ for classification tasks using the `LIBSVM` package interfacing with `MLJ` module.
=#

using DataFrames, CSV
using MLJ

df = CSV.read("../Datasets/Social_Network_Ads.csv", DataFrame)
describe(df)
## Unpacking Data
y, Z = unpack(df,
    ==(:Purchased);     # y is the :Purchased column
    #!=(:Age);          # Z is the rest, except :Age
    :Age => Continuous, # Correcting wrong scitypes
    :EstimatedSalary => Continuous,
    :Purchased => Multiclass)
### Construct an Abstract Matrix `X`
X = hcat(Z.Age, Z.EstimatedSalary)
### Splitting Data into Train and Test
train, test = partition(eachindex(y), 0.8, shuffle=true, rng=123)
# Import SVC and bind it to SVM
SVM = @load SVC pkg = LIBSVM
clf = SVM()
# Train the classifier on the training data
mach = machine(clf, table(X[train, :]), y[train]) |> fit!
# Use the trained classifier to make predictions on the test data
y_hat = predict(mach, X[test, :])
# Evaluate the model's performance
accuracy = mean(y_hat .== y[test]);
println("Accuracy is about $(round(100*accuracy))%")

#=
Note that this is just one way to implement an `SVM`` in _Julia_, and there are many other packages and approaches we can use. In this example, we used the `LIBSVM`` package, which provides a convenient interface for working with `SVM`s in _Julia_.
=#
