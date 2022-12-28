using DataFrames, CSV
using MLJ

df = CSV.read("../Datasets/Social_Network_Ads.csv", DataFrame)
describe(df)
schema(df)
##Unpack Features & Target
y, Z = unpack(df,
    ==(:Purchased);
    # !=(:Age);
    :Age => Continuous,
    :EstimatedSalary => Continuous,
    :Purchased => Multiclass)
X = hcat(Z.Age, Z.EstimatedSalary)
## Split the data into Train and Test Sets
train, test = partition(eachindex(y), 0.8, rng=123)
Xtrain, Xtest = X[train, :], X[test, :]
ytrain, ytest = y[train], y[test]
## Standardizer
sc = Standardizer()
mach_age = machine(sc, Xtrain[:, 1]) |> fit!
Xtrain[:, 1] = MLJ.transform(mach_age, Xtrain[:, 1])
Xtest[:, 1] = MLJ.transform(mach_age, Xtest[:, 1])
mach_salary = machine(sc, Xtrain[:, 2]) |> fit!
Xtrain[:, 2] = MLJ.transform(mach_salary, Xtrain[:, 2])
Xtest[:, 2] = MLJ.transform(mach_salary, Xtest[:, 2])
## Load the `LogisticClassifier` and Bind it to `lc`
LC = @load LogisticClassifier pkg = MLJLinearModels
lc = LC()
## Train the Logistic Classifier
mach = machine(lc, table(Xtrain), ytrain) |> fit!
## Predict the `Xtest`
yhat = predict_mode(mach, Xtest)
## Accuracy
acc = mean( yhat .== ytest);
println("Accuracy is about $(round(100*acc))%")
