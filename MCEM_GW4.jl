using Random
using Plots
using BenchmarkTools
using Distributions
using ArgParse
using Measures
using DelimitedFiles
using LaTeXStrings
using NPZ
using Dates

function parse_commandline()
    settings = ArgParseSettings()

    @add_arg_table! settings begin
        "--phenotypes"
            help = "number of phenotypes"
            arg_type = Int
            default = 4
        "--npar"
            help = "number of transition parameters per phenotype"
            arg_type = Int
            default = 15
        "--generations"
            help = "number of generations"
            arg_type = Int
            default = 6
        "--simulations"
            help = "number of Monte Carlo simulations per EM step"
            arg_type = Int
            default = 3000
        "--steps"
            help = "number of EM steps"
            arg_type = Int
            default = 1
        "--batches"
            help = "number of MCEM batches"
            arg_type = Int
            default = 1
        "--data-file"
            help = "path to input data file"
            arg_type = String
            default = "dataset_hematopoiesis.txt"
        "--data-fraction"
            help = "fraction of dataset rows to use (0 < fraction <= 1)"
            arg_type = Float64
            default = 1.0
        "--output-dir"
            help = "directory where outputs are written"
            arg_type = String
            default = "test_results"
        "--seed"
            help = "set random seed for reproducible runs"
            arg_type = Int
            default = -1
        "--theta-init-file"
            help = "optional path to initial theta matrix file (shape: phenotypes x npar)"
            arg_type = String
            default = ""
    end

    return parse_args(settings)
end

function Galton_watson(θ, ys, y, z, generations, phenotypes, Tₑ, sim, npar)

    # cumulative sum along columns (axis=1 in NumPy)
    Sθ = cumsum(θ, dims=2)

    for gen in 1:(generations-1)   # Julia is 1-based
        # count the number of transitions occurring in a generation
        w = zeros(Int, phenotypes, npar)

        for phenotype in 1:phenotypes
            # current population
            ys[phenotype, gen, sim] = y[phenotype]

            # update transitions for this phenotype
            w, z = update_phenotypic_population(y[phenotype], Sθ, w, z, phenotype)
        end
    

        # update the population (explicit 3-type version)
        Δy1 = sum(Tₑ[1,:,1] .* w[1,:] + Tₑ[1,:,2] .* w[2,:] + Tₑ[1,:,3] .* w[3,:] + Tₑ[1,:,4] .* w[4,:])
        Δy2 = sum(Tₑ[2,:,1] .* w[1,:] + Tₑ[2,:,2] .* w[2,:] + Tₑ[2,:,3] .* w[3,:] + Tₑ[2,:,4] .* w[4,:])
        Δy3 = sum(Tₑ[3,:,1] .* w[1,:] + Tₑ[3,:,2] .* w[2,:] + Tₑ[3,:,3] .* w[3,:] + Tₑ[3,:,4] .* w[4,:])
        Δy4 = sum(Tₑ[4,:,1] .* w[1,:] + Tₑ[4,:,2] .* w[2,:] + Tₑ[4,:,3] .* w[3,:] + Tₑ[4,:,4] .* w[4,:])
        y .+= [Δy1, Δy2, Δy3, Δy4]
    end

    return ys, y, z
end


function update_phenotypic_population(yi, Sθ, w, z, phenotype)
    for c in 1:yi   # loop over cells
        r = rand()
        transition = 1
        done = true

        while done
            if r >= Sθ[phenotype, transition]
                transition += 1
            else
                w[phenotype, transition] += 1
                z[phenotype, transition] += 1
                done = false
            end
        end
    end
    return w, z
end

function Monte_Carlo_Expectation_Maximization(phenotypes, npar, generations, simulations, steps, θ, μ)
    # θs: store θ estimates over EM steps
    θs = ones(Float64, phenotypes, npar, steps+1)
    θs[:, :, 1] .= θ

    Ls = zeros(Float64, steps)

    # loop over EM steps
    for step in 1:steps
        if step % 1 == 0
            println("step: ", step)
        end

        ys = zeros(Int, phenotypes, generations, simulations)
        q  = zeros(Float64, phenotypes, npar)
        τ  = ones(Int, (2^generations+1, 2^generations+1, 2^generations+1, 2^generations+1))
        L  = 0.0

        for sim in 1:simulations
            y = [1, 0, 0, 0]                   # initial population
            z = zeros(Int, phenotypes, npar)   # hidden transitions

            ys, y, z = Galton_watson(θ, ys, y, z, generations, phenotypes, Tₑ, sim, npar)
            #print("y: ", y)

            # Julia: adjust indices (+1) because arrays are 1-based
            i1, i2, i3, i4 = y .+ 1

            # update q with broadcasting
            q .= q .+ (1.0 / sim) .* (z .* τ[i1, i2, i3, i4] .* μ[i1, i2, i3, i4] .- q)

            # update likelihood
            L += (1.0 / sim) * (τ[i1, i2, i3, i4].* μ[i1, i2, i3, i4] - L)

            # increment τ
            τ .+= 1
            τ[i1, i2, i3, i4] = 1
        end

        # M-step: normalize q to update θ
        r = sum(q, dims=2)
        θ .= q ./ r    # broadcasting along columns

        θs[:, :, step+1] .= θ
        Ls[step] = L
    end

    return θs, Ls
end

script_start_ns = time_ns()
parsed_args = parse_commandline()


# ---------------- Parameters ----------------
phenotypes = parsed_args["phenotypes"]
npar = parsed_args["npar"]
generations = parsed_args["generations"]
simulations = parsed_args["simulations"]
steps = parsed_args["steps"]
batches = parsed_args["batches"]
data_file = parsed_args["data-file"]
data_fraction = parsed_args["data-fraction"]
output_dir = parsed_args["output-dir"]
seed = parsed_args["seed"]
theta_init_file = parsed_args["theta-init-file"]

if seed != -1
    Random.seed!(seed)
end

if !(0.0 < data_fraction <= 1.0)
    error("data-fraction must be in (0, 1]. Received data-fraction=$(data_fraction)")
end

# --------------------------------------------------
# Transition effect matrices
# row = type, column = transition

T1 = [-1  0 -1 -1 -1  1  0  0  0 -1 -1 -1 -1 -1 -1;
       0  0  1  0  0  0  1  0  0  2  0  0  1  1  0;
       0  0  0  1  0  0  0  1  0  0  2  0  1  0  1;
       0  0  0  0  1  0  0  0  1  0  0  2  0  1  1]

T2 = vcat(T1[2, :]', T1[1, :]', T1[3, :]', T1[4, :]')  # type 2
T3 = vcat(T2[1, :]', T2[3, :]', T2[2, :]', T2[4, :]')  # type 3
T4 = vcat(T3[1, :]', T3[2, :]', T3[4, :]', T3[3, :]')  # type 4

# 3D array: phenotype × phenotype × transition
Tₑ = cat(T1, T2, T3, T4; dims=3)

# ---------------- Import Data ----------------
data_start_ns = time_ns()
hematopoiesis = Int.(readdlm(data_file))

if ndims(hematopoiesis) != 2 || size(hematopoiesis, 1) != phenotypes
    error("Expected data matrix with $(phenotypes) rows. Got size=$(size(hematopoiesis))")
end

if data_fraction < 1.0
    total_cols = size(hematopoiesis, 2)
    nselected = max(1, floor(Int, total_cols * data_fraction))
    selected = randperm(total_cols)[1:nselected]
    hematopoiesis = hematopoiesis[:, selected]
    println("Using $(nselected)/$(total_cols) data points (fraction=$(data_fraction)).")
end

y1ₒ = hematopoiesis[1,:]   #"LES"
y2ₒ = hematopoiesis[2,:]   #"G2H"
y3ₒ = hematopoiesis[3,:]   #"G1/2H" 
y4ₒ = hematopoiesis[4,:]   #"P1H" 

ndata=size(y1ₒ)[1]
println("Shape of y1ₒ: ", size(y1ₒ))


# ---------------- Compute empirical distribution μ ----------------
# Stack columns
Y = hcat(y1ₒ, y2ₒ, y3ₒ, y4ₒ)

# Find unique outcomes and counts
yₒ = unique(Y, dims=1)
println("Number of unique data: yₒ ", size(yₒ, 1))

# counts
dim = 2^generations + 1   # max cells per phenotype + 1
μ = zeros(Float64, dim, dim, dim, dim)  # 4D frequency array
counts = [sum(all(Y .== yₒ[i, :]', dims=2)) for i in 1:size(yₒ,1)]

# Fill μ
for i in 1:size(yₒ, 1)
    i1, i2, i3, i4 = yₒ[i, :] .+ 1  # Julia indexing
    μ[i1, i2, i3, i4] = counts[i] / sum(counts)
end

data_elapsed_s = (time_ns() - data_start_ns) / 1e9



# ---------------- MCEM algorithm ----------------
results = Vector{Tuple{Array{Float64,3}, Vector{Float64}}}()
batch_mcem_times_s = Float64[]

for batch in 1:batches
    println("batch: ", batch)

    # ------------------- initial conditions -------------------
    O = 0.0 #parameters not in the linear chain model
    X = 0.0 #paramters not in the minimal models

    #α = ones(Float64, npar) ./ npar
    #    0   1   2   3   4   5   6   7   8   9  10  11  12   13  14
    α = [O,  1,  X,  O,  X,  1,  X,  O,  X,  X,  O,  X,  O,  1,  O]  #LES
    β = [O,  1,  X,  X,  O,  1,  1,  X,  O,  X,  X,  O,  1,  O,  O]  #G2H
    γ = [O,  1,  O,  X,  O,  1,  O,  1,  O,  O,  X,  O,  O,  O,  O]  #G1/2H
    δ = [O,  1,  X,  O,  O,  1,  1,  O,  O,  X,  O,  O,  O,  O,  O]  #P1H
    
    
    α = α./sum(α)
    β = β./sum(β)
    γ = γ./sum(γ)
    δ = δ./sum(δ)
    # Stack into θ: (phenotypes × npar)
    θ = vcat(α', β', γ', δ')

    if !isempty(theta_init_file)
        θ_from_file = Float64.(readdlm(theta_init_file))
        if size(θ_from_file) != (phenotypes, npar)
            error("theta init file must have shape $(phenotypes)x$(npar), got $(size(θ_from_file))")
        end
        θ .= θ_from_file
    end

    println(
        "check for normalizations: ",
        sum(θ, dims=2)[:, 1]
    )

    # ----------------------------------------------------------

    mcem_elapsed_s = @elapsed begin
        θs_batch, Ls_batch = Monte_Carlo_Expectation_Maximization(phenotypes, npar, generations, simulations, steps, θ, μ)
        push!(results, (θs_batch, Ls_batch))
    end
    push!(batch_mcem_times_s, mcem_elapsed_s)
end


batch=1

θs, Ls = results[batch]
path = output_dir
mkpath(path)

filename = path * "/θs-GW4_ngen" * string(generations) *
           "_ndata" * string(ndata) *
           "_nsim" * string(simulations) *
           "_batch" * string(batch) * ".npz"

npzwrite(filename, Dict("θs" => θs))

filename = path * "/Ls-GW4_ngen" * string(generations) *
           "_ndata" * string(ndata) *
           "_nsim" * string(simulations) *
           "_batch" * string(batch) * ".npz"

npzwrite(filename, Dict("Ls" => Ls))

total_mcem_elapsed_s = sum(batch_mcem_times_s)
total_elapsed_s = (time_ns() - script_start_ns) / 1e9

timing_filename = path * "/timing-GW4_ngen" * string(generations) *
                 "_ndata" * string(ndata) *
                 "_nsim" * string(simulations) *
                 "_batch" * string(batch) * ".txt"

open(timing_filename, "w") do io
    println(io, "timestamp=" * string(now()))
    println(io, "script=MCEM_GW4-hematopoiesis-model1_script.jl")
    println(io, "phenotypes=" * string(phenotypes))
    println(io, "npar=" * string(npar))
    println(io, "generations=" * string(generations))
    println(io, "simulations=" * string(simulations))
    println(io, "steps=" * string(steps))
    println(io, "batches=" * string(batches))
    println(io, "data_preparation_seconds=" * string(round(data_elapsed_s, digits=6)))
    println(io, "mcem_total_seconds=" * string(round(total_mcem_elapsed_s, digits=6)))
    println(io, "mcem_batch_seconds=" * string(join(round.(batch_mcem_times_s, digits=6), ",")))
    println(io, "total_script_seconds=" * string(round(total_elapsed_s, digits=6)))
end




