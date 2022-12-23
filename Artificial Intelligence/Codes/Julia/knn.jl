using CSV, DataFrames

df = CSV.read("../Datasets/Social_Network_Ads.csv", DataFrame)
x = Float64.(df[!, 2]);
y = df[!, end];

println(typeof(x), size(x))
l = size(x)[1]

using Plots; unicodeplots() 
g1 = scatter(x, y; c=y, legend=false); 

using NearestNeighbors

# KDTree(data, metric; leafsize, reorder)
tree = KDTree(x')
# Initialize k for k-NN
k = 3

tst = rand(1:l, Int(.2*l)) 
# Find nearest neighbors using k-NN and k-d tree
idxs, dists = knn(tree, x[tst], k, true)

