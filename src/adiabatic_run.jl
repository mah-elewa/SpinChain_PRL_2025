# Import necessary packages
using LinearAlgebra
using ITensors, ITensorMPS
using Plots
using LaTeXStrings
using Statistics
using DelimitedFiles
using JLD2

# Define the function to calculate the energy of a given MPS

# This funcion creates the gates for 1 step time evolution of TEBD
function create_gates(s, N, mu, F, tau, periodic=false)
    gates = ITensor[]

    Id = [1 0; 0 1]
    Z = [1 0; 0 -1]
    Sp = [0 1; 0 0]
    Sm = [0 0; 1 0]

    hj_mat = 0.5 * mu * kron(Z,Id) - kron(Sp,Sm) - kron(Sm,Sp) - F * kron(Sp,Sp) - F * kron(Sm,Sm)

    Gj_mat = LinearAlgebra.exp(-im * tau / 2 * hj_mat)

    for j in 1:(N - 1)
        s1 = s[j]
        s2 = s[j + 1]

        Gj = ITensor(Gj_mat, s2, s1, s2', s1')

        push!(gates, Gj)
    end

    # Handle boundary conditions
    if !periodic
        s1 = s[N]
        s2 = s[1]

        hj = mu * op("Sz", s1) * op("Id",s2)

        Gj = exp(-im * tau / 2 * hj)
        push!(gates, Gj)
    else
        s1 = s[N]
        s2 = s[1]

        Gj = ITensor(Gj_mat, s2, s1, s2', s1')

        push!(gates, Gj)
    end

    # Include gates in reverse order
    append!(gates, reverse(gates))

    return gates
end

# This function calculate the density of spins pointing down for any mps
function density_down(s,N,mps)

    spin_up = 0

    for j in 1:N
        spin_up += 2 * expect(mps, "Sz"; sites=j) # 2 because Sz = sigma_z/2
    end

    density = 1 - spin_up / N

    return density

end

# This function calculate the two point correlations of Sz of spins for any mps
function Sz_correlation2(s,N,mps,site1,site2)

    cutoff = 1E-8


    op1 = op("Sz", s[site1])
    op2 = op("Sz", s[site2])

    op3 = op1 * op2

    mps2 = apply(op3, mps; cutoff)

    expval = 4 * inner(mps, mps2) # ⟨psi|sz(site1) sz(site2)|psi⟩


    # Calculate the correlation    
    corr = expval - 4 * expect(mps, "Sz"; sites=site1) * expect(mps, "Sz"; sites=site2)

    return corr

end

# This function calculates the two-point correlation of S+ and S- operators
function SpSm_correlation(s, N, mps, site1, site2)

    cutoff = 1E-8

    op1 = op("S+", s[site1])
    op2 = op("S-", s[site2])

    op_combined = op1 * op2

    mps_applied = apply(op_combined, mps; cutoff)


    expval = inner(mps, mps_applied)


    exp_sp = expect(mps, "S+"; sites=site1)
    exp_sm = expect(mps, "S-"; sites=site2)

    # The correlation function is ⟨S+(site1) S-(site2)⟩ - ⟨S+(site1)⟩⟨S-(site2)⟩
    corr = expval - exp_sp * exp_sm

    return corr

end

# This function runs the simulation for a given set of parameters
function run_simulation(N, i, F, alpha, site)
    # Initialize the MPS with the given parameters   
    cutoff_val = 1E-8 # Cutoff value for the MPS
    s = siteinds("S=1/2", N)
    psi = MPS(s,"Up")

    mu_array = 0.0:0.1:5.0 # Array of detuning values
    mu = mu_array[i] # Select the run value of detuning

    tf_adiabatic = F / alpha # Final time of the adiabatic simulation
    tau = 0.1 # Time step for the simulation

    site1 = site # First site for correlation calculations
    site2 = site + 1 # Second site for correlation calculations

    filename = "output_for_N$(N)_mu$(i).txt" # Output filename

    # Adiabatic time evolution
    step = 0
    for t in 0.0:tau:tf_adiabatic
        Ft = alpha * t
        gates = create_gates(s,N,mu,Ft,tau,true)
        psi = apply(gates, psi; cutoff=cutoff_val)
        normalize!(psi)
        # Print and save every 100 time steps to track progress
        if step % 100 == 0
            bond_dims = linkdims(psi)
            max_bond_dim = maximum(bond_dims)
            open("log_N$(N)_mu$(i).txt", "a") do logf
                println(logf, "Time: $t, Step: $step, Max bond dimensions: $max_bond_dim")
            end
        end
        step += 1
    end

    # Constant F time evolution
    tf_constant_F = tf_adiabatic + 30.0 # Final time for constant F evolution
    results_density = Float64[]
    results_sz_corr = Float64[]
    results_spsm_corr = Float64[]
    
    push!(results_density, real(density_down(s,N,psi)))
    push!(results_sz_corr, real(Sz_correlation2(s,N,psi,site1,site2)))
    push!(results_spsm_corr, real(SpSm_correlation(s,N,psi,site1,site2)))
    
    gates = create_gates(s,N,mu,F,tau,true) # F is now constant

    for t in (tf_adiabatic + tau):tau:tf_constant_F
        
        psi = apply(gates, psi; cutoff=cutoff_val)
        normalize!(psi)

        # Store results for averaging
        push!(results_density, real(density_down(s,N,psi)))
        push!(results_sz_corr, real(Sz_correlation2(s,N,psi,site1,site2)))
        push!(results_spsm_corr, real(SpSm_correlation(s,N,psi,site1,site2)))

        # Print and save every 50 time steps to track progress
        if step % 50 == 0
            bond_dims = linkdims(psi)
            max_bond_dim = maximum(bond_dims)
            open("log_N$(N)_mu$(i).txt", "a") do logf
                println(logf, "Time: $t, Step: $step, Max bond dimensions: $max_bond_dim")
            end
        end
        step += 1
    end
    
    # Calculate averages
    avg_density = mean(results_density)
    avg_sz_corr = mean(results_sz_corr)
    avg_spsm_corr = mean(results_spsm_corr)

    final_results = [avg_density, avg_sz_corr, avg_spsm_corr]
    
    writedlm(filename, final_results)
end

# Define variables for the simulation
N = 40 # Number of sites in the chain
i = parse(Int, ARGS[1]) # Get the index of the run from command line arguments

# Define parameters for the adiabatic evolution
F = 0.5 # Final value of the drive
alpha = 0.001 # Adiabatic parameter

# Select the site indices for correlation calculations
site = trunc(Int, N/2)  

# Run the simulation
@time begin
    run_simulation(N, i, F, alpha, site)
end
