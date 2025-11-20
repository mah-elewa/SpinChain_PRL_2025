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

    density = 1 - spin_up /  N

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

# This function calculate the two point correlations of Sx of spins for any mps
function Sx_correlation2(s,N,mps,site1,site2)

    cutoff = 1E-8


    op1 = op("Sx", s[site1])
    op2 = op("Sx", s[site2])

    op3 = op1 * op2

    mps2 = apply(op3, mps; cutoff)

    expval = 4 * inner(mps, mps2) # ⟨psi|sx(site1) sx(site2)|psi⟩


    # Calculate the correlation    
    # Calculate the correlation    
    corr = expval - 4 * expect(mps, "Sx"; sites=site1) * expect(mps, "Sx"; sites=site2)

    return corr

end

# This function calculate the two point correlations of Sy of spins for any mps
function Sy_correlation2(s,N,mps,site1,site2)

    cutoff = 1E-8


    op1 = op("Sy", s[site1])
    op2 = op("Sy", s[site2])

    op3 = op1 * op2

    mps2 = apply(op3, mps; cutoff)

    expval = 4 * inner(mps, mps2) # ⟨psi|sy(site1) sy(site2)|psi⟩


    # Calculate the correlation
    corr = expval + (expect(mps, "S+"; sites=site1) - expect(mps, "S-"; sites=site1)) * (expect(mps, "S+"; sites=site2) - expect(mps, "S-"; sites=site2))
    return corr

end

# This function calculate the three point correlations of Sz of spins for any mps
function Sz_correlation3(s,N,mps,site1,site2,site3)

    cutoff = 1E-8


    op1 = op("Sz", s[site1])
    op2 = op("Sz", s[site2])
    op3 = op("Sz", s[site3])

    op4 = op1 * op2 * op3

    mps2 = apply(op4, mps; cutoff)


    expval = 8 * inner(mps, mps2) # ⟨psi|sz(site1) sz(site2) sz(site3)|psi⟩


    # Calculate the correlation    
    corr = expval - 8 * expect(mps, "Sz"; sites=site1) * expect(mps, "Sz"; sites=site2) * expect(mps, "Sz"; sites=site3)

    return corr

end

# This function runs the simulation for a given set of parameters
function run_simulation(N, i, F, tf, t_avg, site)
    
    # Initialize the MPS with the given parameters   
    cutoff_val = 1E-8 # Cutoff value for the MPS
    s = siteinds("S=1/2", N)
    psi = MPS(s,"Up")

    mu_array = 0.0:0.1:5.0 # Array of detuning values
    mu = mu_array[i] # Select the run value of detuning

    results = [0.0,0.0,0.0,0.0,0.0] # Initialize results array

    tau = 0.1 # Time step for the simulation

    site1 = site # First site for correlation calculations
    site2 = site + 1 # Second site for correlation calculations
    site3 = site + 2 # Third site for correlation calculations

    filename = "sudden_output_for_mu$(i).txt" # Output filename
    
    gates = create_gates(s,N,mu,F,tau,true)

    # time evolution
    step = 0
    avg_steps = 0
    for t in 0.0:tau:tf
        # Time evolution step
        psi = apply(gates, psi; cutoff=cutoff_val)
        normalize!(psi)

        # Print and save every 50 time steps
        if step % 50 == 0
            # Print bond dimensions
            bond_dims = linkdims(psi)
            max_bond_dim = maximum(bond_dims)
            open("log_mu$(i).txt", "a") do logf
                println(logf, "Time: $t, Step: $step, Max bond dimensions: $max_bond_dim")
            end

        end
        step += 1

        # Average the results over the specified time interval
        if t > (tf-t_avg)
            results[1] += real(density_down(s,N,psi)) # Accumulate the density of spins pointing down
            results[2] += real(Sz_correlation2(s,N,psi,site1,site2)) # Accumulate Sz correlation
            results[3] += real(Sx_correlation2(s,N,psi,site1,site2)) # Accumulate Sx correlation
            results[4] += real(Sy_correlation2(s,N,psi,site1,site2)) # Accumulate Sy correlation
            results[5] += real(Sz_correlation3(s,N,psi,site1,site2,site3)) # Accumulate Sz correlation 3
            
            avg_steps += 1
        end
    end

    results[1] = results[1] / avg_steps # Store the density of spins pointing down
    results[2] = results[2] / avg_steps # Store Sz correlation
    results[3] = results[3] / avg_steps # Store Sx correlation
    results[4] = results[4] / avg_steps # Store Sy correlation
    results[5] = results[5] / avg_steps # Store Sz correlation 3

    writedlm(filename, results)

end


# Define variables for the simulation

N = 16 # Number of sites in the chain

i = parse(Int, ARGS[1]) # Convert argument to integer  # Get the index of the run from command line arguments

# Define parameters for the adiabatic evolution
F = 0.5 # Final value of the drive

tf = 50.0 # Final time for the simulation
t_avg = 30.0 # Time interval for averaging results

# Select the site indices for correlation calculations
site = trunc(Int, N/2)  


@time begin

    run_simulation(N, i, F, tf, t_avg, site)
end

