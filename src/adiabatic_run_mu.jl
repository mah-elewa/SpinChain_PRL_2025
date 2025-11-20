# Import necessary packages
using LinearAlgebra
using ITensors, ITensorMPS
using Statistics
using DelimitedFiles
using JLD2
using Base.Threads # For multithreading

# Global lock for thread-safe logging to the main log file
const LOG_LOCK = ReentrantLock()

# This function creates the gates for 1 step time evolution of TEBD
function create_gates(s, N, mu, F, tau, periodic=false)
    gates = ITensor[]

    Id = [1 0; 0 1]
    Z = [1 0; 0 -1]
    Sp = [0 1; 0 0]
    Sm = [0 0; 1 0]

    # Hamiltonian matrix for a pair of sites
    hj_mat = 0.5 * mu * kron(Z,Id) - kron(Sp,Sm) - kron(Sm,Sp) - F * kron(Sp,Sp) - F * kron(Sm,Sm)

    # Two-site gate from the Hamiltonian
    Gj_mat = LinearAlgebra.exp(-im * tau / 2 * hj_mat)

    for j in 1:(N - 1)
        s1 = s[j]
        s2 = s[j + 1]
        Gj = ITensor(Gj_mat, s2, s1, s2', s1')
        push!(gates, Gj)
    end

    # Handle boundary conditions (periodic or open)
    if !periodic
        # This part of the code was in your original, likely for specific open boundary Hamiltonians.
        # Given `periodic=true` is used in run_simulation, this block is typically not executed.
        s1 = s[N]
        s2 = s[1] # s2 is not directly involved in hj, but makes it a 2-site operator
        hj = mu * op("Sz", s1) * op("Id",s2) 
        Gj = exp(-im * tau / 2 * hj)
        push!(gates, Gj)
    else
        # For periodic boundary conditions, add a gate connecting N and 1
        s1 = s[N]
        s2 = s[1]
        Gj = ITensor(Gj_mat, s2, s1, s2', s1')
        push!(gates, Gj)
    end

    # Apply gates in forward and reverse order for symmetric Trotter decomposition
    append!(gates, reverse(gates))

    return gates
end

# This function calculates the density of spins pointing down for any MPS
function density_down(s,N,mps)
    spin_up = 0
    for j in 1:N
        spin_up += 2 * expect(mps, "Sz"; sites=j) # 2 because Sz = sigma_z/2
    end
    density = 1 - spin_up / N
    return density
end

# This function calculates the two-point correlations of Sz of spins for any MPS
function Sz_correlation2(s,N,mps,site1,site2)
    cutoff = 1E-8
    op1 = op("Sz", s[site1])
    op2 = op("Sz", s[site2])
    op3 = op1 * op2
    mps2 = apply(op3, mps; cutoff)
    expval = 4 * inner(mps, mps2) # ⟨psi|sz(site1) sz(site2)|psi⟩
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
    corr = expval - exp_sp * exp_sm
    return corr
end

# Helper function for the constant mu averaging phase, to be spawned as a thread
function perform_averaging_task(psi_start, s, N, mu_val, F_const, tau, cutoff_val, site1, site2, results_output_filename, log_filename, avg_task_id)
    psi_copy = deepcopy(psi_start) # Work on a copy of the MPS
    
    results_density = Float64[]
    results_sz_corr = Float64[]
    results_spsm_corr = Float64[]

    # Perform the first measurement at the start of the averaging period
    push!(results_density, real(density_down(s,N,psi_copy)))
    push!(results_sz_corr, real(Sz_correlation2(s,N,psi_copy,site1,site2)))
    push!(results_spsm_corr, real(SpSm_correlation(s,N,psi_copy,site1,site2)))

    # Pre-calculate gates for constant mu evolution (more efficient if gates don't change)
    gates_avg = create_gates(s,N,mu_val,F_const,tau,true)

    # Time evolution for 30 units with constant mu_val and F_const
    # Starting t_avg from tau because the initial state is already measured
    for t_avg in tau:tau:30.0 
        psi_copy = apply(gates_avg, psi_copy; cutoff=cutoff_val)
        normalize!(psi_copy)

        push!(results_density, real(density_down(s,N,psi_copy)))
        push!(results_sz_corr, real(Sz_correlation2(s,N,psi_copy,site1,site2)))
        push!(results_spsm_corr, real(SpSm_correlation(s,N,psi_copy,site1,site2)))
    end
    
    # Calculate averages
    avg_density = mean(results_density)
    avg_sz_corr = mean(results_sz_corr)
    avg_spsm_corr = mean(results_spsm_corr)

    final_results = [avg_density, avg_sz_corr, avg_spsm_corr]
    
    # Save results to a file specific to the mu_val
    writedlm(results_output_filename, final_results)
    
    # Log completion message to the main log file (thread-safe)
    lock(LOG_LOCK) do
        open(log_filename, "a") do logf
            println(logf, "Averaging task (ID $avg_task_id) for mu=$(mu_val) completed. Results saved to $results_output_filename")
        end
    end
    return nothing # Task doesn't return a value, just performs side effects (saving file, logging)
end


# This function runs the entire simulation for a given set of parameters
function run_simulation(N, F_final, alpha, site)
    cutoff_val = 1E-8 # Cutoff value for the MPS
    s = siteinds("S=1/2", N)
    psi = MPS(s,"Up") # Initial state for the entire simulation (F=0, mu=0)

    tau = 0.1 # Time step for the simulation

    site1 = site # First site for correlation calculations
    site2 = site + 1 # Second site for correlation calculations

    log_filename = "log_N$(N)_F$(F_final)_alpha$(alpha).txt"
    
    # Clear previous log file content for this run
    open(log_filename, "w") do logf
        println(logf, "Starting simulation N=$N, F_final=$F_final, alpha=$alpha")
    end

    # --- Phase 1: Adiabatic Ramp of F (mu fixed at 5.0) ---
    initial_mu_F_ramp = 5.0 # Fixed mu value during F ramp
    tf_F_ramp = F_final / alpha # Final time for F ramp (when F reaches F_final)

    lock(LOG_LOCK) do
        open(log_filename, "a") do logf
            println(logf, "\n--- Phase 1: Adiabatic F ramp (mu fixed at $initial_mu_F_ramp) ---")
        end
    end

    global_step = 0 # Global step counter for all evolution
    for t_F_ramp in 0.0:tau:tf_F_ramp
        global_step += 1
        current_F_ramp = alpha * t_F_ramp # F ramps linearly from 0 to F_final
        gates = create_gates(s,N,initial_mu_F_ramp,current_F_ramp,tau,true)
        psi = apply(gates, psi; cutoff=cutoff_val)
        normalize!(psi)

        if global_step % 50 == 0 # Log every 50 steps
            bond_dims = linkdims(psi)
            max_bond_dim = maximum(bond_dims)
            lock(LOG_LOCK) do # Use lock for thread-safe logging
                open(log_filename, "a") do logf
                    println(logf, "F-Ramp Step: $global_step, Time: $(round(t_F_ramp, digits=3)), Current F: $(round(current_F_ramp, digits=3)), Max bond dimensions: $max_bond_dim")
                end
            end
        end
    end
    # Ensure psi is the state at the exact end of F-ramp
    # (The loop handles the last step, so psi should be correct)


    # --- Phase 2: Mu Ramp-Down with Parallel Averaging ---
    const_F_mu_ramp = F_final # F is now constant at its final value (0.5)

    # Mu values where averaging needs to be triggered
    # Convert to Set for efficient `in` checks and deletions
    mu_averaging_points_set = Set(collect(0.0:0.1:5.0))
    
    # Store pending averaging tasks
    averaging_tasks = []
    avg_task_counter = 0 # Unique ID for each averaging task

    # Current mu value during the main ramp-down. Starts from the initial_mu_F_ramp value.
    current_mu_main_ramp = initial_mu_F_ramp 

    lock(LOG_LOCK) do
        open(log_filename, "a") do logf
            println(logf, "\n--- Phase 2: Mu ramp-down with parallel averaging (F fixed at $const_F_mu_ramp) ---")
        end
    end

    # Handle the very first averaging point (mu=5.0), which is the state after F-ramp.
    # We use a small tolerance for floating point comparison.
    tolerance = 1e-9 
    if current_mu_main_ramp in mu_averaging_points_set || isapprox(current_mu_main_ramp, 5.0, atol=tolerance)
        avg_task_counter += 1
        # Round mu for filename
        output_file = "output_N$(N)_mu$(round(5.0, digits=1)).txt" 
        
        lock(LOG_LOCK) do
            open(log_filename, "a") do logf
                println(logf, "Scheduling averaging task (ID $avg_task_counter) for mu=5.0 at overall step $global_step.")
            end
        end
        
        # Spawn a new thread to perform the averaging
        task = Threads.@spawn perform_averaging_task(deepcopy(psi), s, N, 5.0, const_F_mu_ramp, tau, cutoff_val, site1, site2, output_file, log_filename, avg_task_counter)
        push!(averaging_tasks, task)
        delete!(mu_averaging_points_set, 5.0) # Mark as scheduled
    end

    # Loop for the continuous mu ramp-down
    # The `t_current_overall` continues from where the F-ramp left off.
    # `t_mu_ramp_progress` is the time elapsed *since* the mu ramp started (i.e., since F-ramp ended).
    t_mu_ramp_progress = 0.0 
    total_mu_ramp_time = 5.0 / alpha # Time to ramp mu from 5.0 to 0.0

    while current_mu_main_ramp > 0.0 - tolerance # Continue until mu reaches 0.0
        global_step += 1
        t_mu_ramp_progress += tau
        
        # Calculate the next mu for the main ramp
        current_mu_main_ramp = 5.0 - alpha * t_mu_ramp_progress
        
        # Ensure mu doesn't go below 0.0 due to small numerical errors
        if current_mu_main_ramp < 0.0
            current_mu_main_ramp = 0.0
        end

        # Apply evolution gate based on current_mu_main_ramp
        gates = create_gates(s,N,current_mu_main_ramp,const_F_mu_ramp,tau,true)
        psi = apply(gates, psi; cutoff=cutoff_val)
        normalize!(psi)

        if global_step % 50 == 0 # Log every 50 steps
            bond_dims = linkdims(psi)
            max_bond_dim = maximum(bond_dims)
            lock(LOG_LOCK) do # Use lock for thread-safe logging
                open(log_filename, "a") do logf
                    println(logf, "Mu-Ramp Step: $global_step, Time in Mu-Ramp: $(round(t_mu_ramp_progress, digits=3)), Current mu: $(round(current_mu_main_ramp, digits=3)), Max bond dimensions: $max_bond_dim")
                end
            end
        end

        # Check if current_mu_main_ramp has passed a mu_averaging_point that hasn't been scheduled yet
        # We iterate through the set of points still to average.
        # Create a temporary list to avoid modifying the set while iterating
        mu_points_to_check = collect(mu_averaging_points_set) 
        for mu_target in mu_points_to_check
            # If current_mu has just crossed or is very close to mu_target
            # (Note: we are checking from higher mu to lower mu, so <= is appropriate)
            if current_mu_main_ramp <= mu_target + tolerance 
                avg_task_counter += 1
                output_file = "output_N$(N)_mu$(round(mu_target, digits=1)).txt"
                
                lock(LOG_LOCK) do
                    open(log_filename, "a") do logf
                        println(logf, "Scheduling averaging task (ID $avg_task_counter) for mu=$(mu_target) at overall step $global_step.")
                    end
                end
                
                # Spawn a new thread to perform the averaging
                task = Threads.@spawn perform_averaging_task(deepcopy(psi), s, N, mu_target, const_F_mu_ramp, tau, cutoff_val, site1, site2, output_file, log_filename, avg_task_counter)
                push!(averaging_tasks, task)
                delete!(mu_averaging_points_set, mu_target) # Mark as scheduled
            end
        end
    end

    # Final check to ensure 0.0 is scheduled if not already
    if 0.0 in mu_averaging_points_set # This handles the case where mu reaches exactly 0.0
         avg_task_counter += 1
        output_file = "output_N$(N)_mu$(round(0.0, digits=1)).txt" 
        
        lock(LOG_LOCK) do
            open(log_filename, "a") do logf
                println(logf, "Scheduling final averaging task (ID $avg_task_counter) for mu=0.0 at overall step $global_step.")
            end
        end
        
        task = Threads.@spawn perform_averaging_task(deepcopy(psi), s, N, 0.0, const_F_mu_ramp, tau, cutoff_val, site1, site2, output_file, log_filename, avg_task_counter)
        push!(averaging_tasks, task)
        delete!(mu_averaging_points_set, 0.0) # Mark as scheduled
    end


    # Wait for all parallel averaging tasks to complete
    lock(LOG_LOCK) do
        open(log_filename, "a") do logf
            println(logf, "\nMain mu-ramp evolution finished. Waiting for $(length(averaging_tasks)) averaging tasks to complete...")
        end
    end
    wait.(averaging_tasks) # Wait for all tasks to finish

    lock(LOG_LOCK) do
        open(log_filename, "a") do logf
            println(logf, "All averaging tasks completed. Simulation finished.")
        end
    end
end


# --- Main execution block ---
# Define variables for the simulation
N = 40 # Number of sites in the chain
F_final = 0.5 # Final value of the drive (F will ramp to this, then stay constant)
alpha = 0.001 # Adiabatic parameter for both F and mu ramps

# Select the site indices for correlation calculations
site = trunc(Int, N/2)  

# Run the simulation
# To enable multithreading, start Julia with `julia -t auto your_script_name.jl`
# or `JULIA_NUM_THREADS=X julia your_script_name.jl`
@time begin
    run_simulation(N, F_final, alpha, site)
end