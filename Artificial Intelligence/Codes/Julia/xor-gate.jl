using Flux

epochs = 20

x = -1:.1:1
n = length(x)
x1 = x2 = Array{Float64}[];
for i ∈ 1:n
	x1 = vcat(x1, x[i].*ones(n, 1))
	x2 = vcat(x2, x)
end

X = hcat(x1, x2)
y = ones(n^2, 1)

vscodedisplay(X, "X")
vscodedisplay(y, "y")

using Plots; gr()

for lim ∈ n:n:n^2
	@show lim
	sc = scatter!(x1[lim-n+1:lim], x2[lim-n+1:lim], legend=:none)
	display(sc)
	sleep(1)
end

# Create the dataset for an "XOR" problem
X = hcat(digits.(0:3, base=2, pad=2)...)
y = reshape(xor.(eachrow(x)...), 1, 4)
data = Flux.Data.DataLoader((X, y))

# `mdl` is the model to be built 
mdl = Chain(
			Dense(2 => 2, relu),
			Dense(2 => 1, σ)
			)
# `ps` contains all trainable parameters			
ps = Flux.params(mdl)
# Predicted output 
ŷ = mdl(x)
# `mloss` is the loss function to be minimized
loss(x, y) = Flux.Losses.logitbinarycrossentropy(mdl(x), y)
# `opt` designates the optimizer
opt = Adam()
#= TRAINING =#
for _ ∈ 1:epochs
	Flux.train!(loss, ps, data, opt)
end