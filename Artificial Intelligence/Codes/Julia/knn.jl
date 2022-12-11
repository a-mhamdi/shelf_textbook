Here is an example of KNN implemented in Julia:

using Distances
using StatsBase

function knn(X::Array{T, 2}, y::Array{U, 1}, x::Array{T, 1}, k::Int) where {T <: Real, U}
    # Calculate distances between x and each point in X
    dists = pairwise(Euclidean(), X, x)

    # Sort the distances and indices in ascending order
    sorted_dists = sortperm(dists)

    # Take the top k distances and their corresponding y values
    y_neighbors = y[sorted_dists[1:k]]

    # Return the majority vote of the neighbors
    return mode(y_neighbors)
end


This implementation uses the Distances and StatsBase packages to calculate distances and perform a majority vote. It takes as input the training data X and labels y, the test point x, and the number of nearest neighbors k, and returns the predicted label for x.
