using Random
using Plots
using BenchmarkTools
using Distributions
using Measures

function Galton_watson(θ, ys, y, z, generations, phenotypes, Tₑ, sim, npar)

    # cumulative sum along columns (axis=1 in NumPy)
    Sθ = cumsum(θ, dims=2)

    for gen in 1:(generations)   # Julia is 1-based
        # count the number of transitions occurring in a generation
        w = zeros(Int, phenotypes, npar)

        for phenotype in 1:phenotypes
            # current population
            ys[phenotype, gen, sim] = y[phenotype]

            # update transitions for this phenotype
            w, z = update_phenotypic_population(y[phenotype], Sθ, w, z, phenotype)
        end
    

        # update the population (explicit 3-type version)
        y1 = sum(Tₑ[1,:,1] .* w[1,:] + Tₑ[1,:,2] .* w[2,:] + Tₑ[1,:,3] .* w[3,:] + Tₑ[1,:,4] .* w[4,:])
        y2 = sum(Tₑ[2,:,1] .* w[1,:] + Tₑ[2,:,2] .* w[2,:] + Tₑ[2,:,3] .* w[3,:] + Tₑ[2,:,4] .* w[4,:])
        y3 = sum(Tₑ[3,:,1] .* w[1,:] + Tₑ[3,:,2] .* w[2,:] + Tₑ[3,:,3] .* w[3,:] + Tₑ[3,:,4] .* w[4,:])
        y4 = sum(Tₑ[4,:,1] .* w[1,:] + Tₑ[4,:,2] .* w[2,:] + Tₑ[4,:,3] .* w[3,:] + Tₑ[4,:,4] .* w[4,:])
        y .+= [y1, y2, y3, y4]
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

phenotypes = 4 # number of phenotypes
npar = 15 # parameters per phenotype

# parameters
α = [0.1, 0.2, 0.0, 0.0, 0.0, 0.7, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
β = [0.1, 0.0, 0.2, 0.0, 0.0, 0.7, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
γ = [0.1, 0.0, 0.0, 0.2, 0.0, 0.7, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
δ = [0.1, 0.7, 0.0, 0.0, 0.0, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

# θ matrix: rows = phenotypes, columns = parameters
θ = vcat(α', β', γ', δ')

println("check for normalizations: ",cumsum(θ, dims=2)[:, end])

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

# --------------------------------------------------
# Simulation parameters
generations  = 6
simulations  = 100

ys = zeros(Int, phenotypes, generations, simulations)

for sim in 1:simulations
    y = [1, 2, 1, 1]                    # initial population
    z = zeros(Int, phenotypes, npar)   # transition counts

    ys, y, z = Galton_watson(θ, ys, y, z, generations, phenotypes, Tₑ, sim, npar)
end


# create a 1×2 layout
p1 = plot(ys[1, :, :], color = :red, xlabel = "generation", ylabel = "number of 1-type cells", legend = false)
p2 = plot(ys[2, :, :], color = :blue,  xlabel = "generation", ylabel = "number of 2-type cells", legend = false)
p3 = plot(ys[3, :, :], color = :green,  xlabel = "generation", ylabel = "number of 3-type cells", legend = false)
p4 = plot(ys[4, :, :], color = :magenta,  xlabel = "generation", ylabel = "number of 4-type cells", legend = false)

plot(p1, p2, p3, p4, layout = (1,4), size = (1000,300), margin = 5mm)


