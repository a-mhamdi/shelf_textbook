using Flux

epochs = 20

#=
x = -1:.1:1
n = length(x)
x1 = x2 = Array{Float64}[];
for i ∈ 1:n
	x1 = vcat(x1, x[i].*ones(n, 1))
	x2 = vcat(x2, x)
end

X = hcat(x1, x2)
=#

# Create the dataset for an "XOR" problem
X = 2 .* rand(Float32, 1000, 2) .- 1;
vscodedisplay(X, "X")

y = [xor(row[1]>0, row[2]>0) for row ∈ eachrow(X)]
vscodedisplay(y, "y")

using Plots
scatter(X[:, 1], X[:, 2], group=y)

data = Flux.Data.DataLoader((X', y'))

# `mdl` is the model to be built 
mdl = Chain(
			Dense(2 => 2, relu),
			Dense(2 => 1, σ)
			)

# `opt` designates the optimizer
opt = Adam()
# `state` contains all trainable parameters
state = Flux.setup(opt, mdl)		

#= TRAINING PHASE=#
losses = []
using ProgressMeter

@showprogress for epoch in 1:10
    for (X, y) in data
		# Begin a gradient context session
        loss, grads = Flux.withgradient(mdl) do m
            # Evaluate model:
            ŷ = m(X)
			# Evaluate loss:
            Flux.binarycrossentropy(ŷ, y)
        end
        Flux.update!(state, mdl, grads[1])
        push!(losses, loss)  # Logging, outside gradient context
    end
end