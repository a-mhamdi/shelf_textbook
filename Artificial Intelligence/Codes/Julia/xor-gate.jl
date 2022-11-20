using Flux
using CUDA

epochs = 2
# Create the dataset for an "XOR" problem
x = hcat(digits.(0:3, base=2, pad=2)...) |> gpu 
y = Flux.onehotbatch(xor.(eachrow(x)...), 0:1) |> gpu
data = ((Float32.(x), y) for _ in 1:100)

# `mdl` is the model to be built 
mdl = Chain(Dense(2 => 3, sigmoid),
	      BatchNorm(3), 
	      Dense(3 => 2)) |> gpu
	      
ps = Flux.params(mdl) # `ps` gathers all the parameters
opt = Adam()

# `mloss` is the loss function to be minimized
mloss(x, y) = Flux.logitcrossentropy(mdl(x), y)

for _ in 1:epochs
 Flux.train!(mloss, ps, data, opt)
end

