K-means is a clustering algorithm that is used to partition a dataset into a specified number of clusters. Here is an example of K-means implemented in Julia:


using Clustering

function kmeans(X::Array{T, 2}, k::Int) where {T <: Real}
	# Initialize cluster centers randomly
	centers = zeros(k, size(X, 2))
	for i in 1:k
		centers[i, :] = X[rand(1:size(X, 1)), :]
		end

		# Repeat until convergence
		converged = false
		while !converged
		# Calculate distances between each point and each cluster center
		dists = pairwise(Euclidean(), X, centers)

		# Assign each point to the closest cluster center
		clusters = argmin(dists, dims=1)

		# Calculate the new cluster centers as the mean of all points in the cluster
		new_centers = zeros(k, size(X, 2))
		for i in 1:k
			if sum(clusters .== i) > 0
				new_centers[i, :] = mean(X[clusters .== i, :], dims=1)
			else
				# If a cluster is empty, randomly initialize a new center
				new_centers[i, :] = X[rand(1:size(X, 1)), :]
			end
		end

		# Check for convergence
		converged = isapprox(centers, new_centers, rtol=1e-6)

		# Update the cluster centers
		centers = new_centers
	end

	return centers, clusters
end



This implementation uses the Clustering package to calculate distances. It takes as input the dataset X and the number of clusters k, and returns the cluster centers and the cluster assignments of each point.


