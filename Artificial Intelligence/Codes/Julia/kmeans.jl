# `K-means` is a clustering algorithm that is used to partition an unlabeled dataset into a specified number of clusters.

using CSV, DataFrames

df = CSV.read("../Datasets/Mall_Customers.csv", DataFrame);
first(df, 5)
income = df[!, 4];
ss = df[!, 5];

using Plots
scatter(income, ss, legend=false)

using Clustering

# FEATURES
X = hcat(ss, income);
typeof(X)
hat_clusters = kmeans(X', 5; display=:iter)

scatter(ss, income, marker_z=hat_clusters.assignments, color=:lightrainbow, legend=false)
scatter!(hat_clusters.centers[1,:], hat_clusters.centers[2,:], color=:black, legend=true)

