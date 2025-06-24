"""
Complete debugging script to identify the exact problem
"""
import numpy as np
import sys
import os
import matplotlib.pyplot as plt

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def minimal_finite_volume_solver():
    """
    Minimal working finite volume solver from scratch
    to compare with the main implementation
    """
    print("="*60)
    print("MINIMAL FINITE VOLUME SOLVER - REFERENCE IMPLEMENTATION")
    print("="*60)
    
    # Simple 2D Poisson: -Δu = f with u = 0 on boundary
    # Domain [0,1] × [0,1]
    nx, ny = 10, 10
    Lx, Ly = 1.0, 1.0
    dx, dy = Lx/nx, Ly/ny
    
    print(f"Mesh: {nx}×{ny}, dx={dx:.4f}, dy={dy:.4f}")
    
    # Create mesh centers
    x = np.linspace(dx/2, Lx-dx/2, nx)
    y = np.linspace(dy/2, Ly-dy/2, ny)
    X, Y = np.meshgrid(x, y, indexing='ij')
    
    # Source function: f(x,y) = 2 (constant)
    # Exact solution: u = x(1-x)y(1-y) satisfies -Δu = 2[x(1-x) + y(1-y)] ≈ 2
    f = 2.0 * np.ones((nx, ny))
    u_exact = X * (1 - X) * Y * (1 - Y)
    
    print(f"Source function: f = 2 (constant)")
    print(f"Exact solution range: [{np.min(u_exact):.6f}, {np.max(u_exact):.6f}]")
    
    # Build system matrix A and RHS b
    n_total = nx * ny
    A = np.zeros((n_total, n_total))
    b = np.zeros(n_total)
    
    def get_index(i, j):
        """Convert (i,j) to global index"""
        return i * ny + j
    
    # Assemble finite volume equations
    for i in range(nx):
        for j in range(ny):
            idx = get_index(i, j)
            
            # Check if boundary cell
            is_boundary = (i == 0 or i == nx-1 or j == 0 or j == ny-1)
            
            if is_boundary:
                # Dirichlet BC: u = 0
                A[idx, idx] = 1.0
                b[idx] = 0.0
            else:
                # Interior cell: finite volume discretization
                # -Δu = -(∂²u/∂x² + ∂²u/∂y²)
                # FV: (u[i-1,j] - 2*u[i,j] + u[i+1,j])/dx² + (u[i,j-1] - 2*u[i,j] + u[i,j+1])/dy²
                
                # Coefficients
                coeff_center = 2.0/dx + 2.0/dy
                coeff_x = -1.0/dx
                coeff_y = -1.0/dy
                
                # Center
                A[idx, idx] = coeff_center
                
                # Neighbors
                A[idx, get_index(i-1, j)] = coeff_x  # West
                A[idx, get_index(i+1, j)] = coeff_x  # East
                A[idx, get_index(i, j-1)] = coeff_y  # South
                A[idx, get_index(i, j+1)] = coeff_y  # North
                
                # RHS
                b[idx] = f[i, j]
    
    print(f"Matrix condition number: {np.linalg.cond(A):.2e}")
    print(f"Matrix determinant: {np.linalg.det(A):.2e}")
    
    # Solve
    u_vec = np.linalg.solve(A, b)
    u_numerical = u_vec.reshape((nx, ny))
    
    print(f"Numerical solution range: [{np.min(u_numerical):.6f}, {np.max(u_numerical):.6f}]")
    
    # Compute error
    error = np.abs(u_numerical - u_exact)
    l2_error = np.sqrt(np.mean(error**2))
    linf_error = np.max(error)
    
    print(f"L2 error: {l2_error:.6e}")
    print(f"L∞ error: {linf_error:.6e}")
    
    if l2_error < 1e-2:
        print("✅ Reference implementation works correctly!")
    else:
        print("❌ Even reference implementation has issues")
    
    return u_numerical, u_exact, error

def debug_main_solver():
    """Debug the main solver step by step"""
    print("\n" + "="*60)
    print("DEBUGGING MAIN SOLVER")
    print("="*60)
    
    # Import main solver components
    from config.config import ProblemConfig
    from src.mesh.mesh_generator import create_mesh
    from src.functions.source_functions import create_source_function, get_boundary_values
    from src.boundary.boundary_conditions import create_boundary_condition
    from src.solver.finite_volume_solver import FiniteVolumeSolverFactory
    
    # Create configuration
    config = ProblemConfig()
    config.mesh.nx = 10
    config.mesh.ny = 10
    config.function_name = "manufactured"
    config.boundary.bc_type = "dirichlet"
    config.solver.solver_type = "direct"
    
    print(f"Configuration: {config.mesh.nx}×{config.mesh.ny}, {config.function_name}")
    
    # Step 1: Check mesh
    mesh = create_mesh(config.mesh, config.domain)
    print(f"\n1. MESH CHECK:")
    print(f"   nx={mesh.nx}, ny={mesh.ny}")
    print(f"   Domain: [{mesh.x_min}, {mesh.x_max}] × [{mesh.y_min}, {mesh.y_max}]")
    print(f"   dx={mesh.dx:.6f}, dy={mesh.dy:.6f}")
    print(f"   Cell centers range: x=[{np.min(mesh.x_centers):.3f}, {np.max(mesh.x_centers):.3f}]")
    print(f"                       y=[{np.min(mesh.y_centers):.3f}, {np.max(mesh.y_centers):.3f}]")
    
    # Step 2: Check source function
    source_function = create_source_function(config.function_name)
    print(f"\n2. SOURCE FUNCTION CHECK:")
    
    # Test at center
    x_test, y_test = 0.5, 0.5
    f_val = source_function.evaluate(x_test, y_test)
    u_exact_val = source_function.exact_solution(x_test, y_test)
    print(f"   At (0.5, 0.5): f = {f_val:.6f}, u_exact = {u_exact_val:.6f}")
    
    # Expected for manufactured: f = 2*(0.5*0.5 + 0.5*0.5) = 1.0, u = 0.5*0.5*0.5*0.5 = 0.0625
    
    # Test on full mesh
    f_mesh = source_function.evaluate(mesh.x_centers, mesh.y_centers)
    u_exact_mesh = source_function.exact_solution(mesh.x_centers, mesh.y_centers)
    print(f"   Source range: [{np.min(f_mesh):.6f}, {np.max(f_mesh):.6f}]")
    print(f"   Exact solution range: [{np.min(u_exact_mesh):.6f}, {np.max(u_exact_mesh):.6f}]")
    
    # Step 3: Check boundary values
    boundary_values = get_boundary_values(source_function, mesh, config.boundary)
    left_vals, right_vals, bottom_vals, top_vals = boundary_values
    print(f"\n3. BOUNDARY VALUES CHECK:")
    print(f"   Left boundary: min={np.min(left_vals):.6f}, max={np.max(left_vals):.6f}")
    print(f"   Right boundary: min={np.min(right_vals):.6f}, max={np.max(right_vals):.6f}")
    print(f"   Bottom boundary: min={np.min(bottom_vals):.6f}, max={np.max(bottom_vals):.6f}")
    print(f"   Top boundary: min={np.min(top_vals):.6f}, max={np.max(top_vals):.6f}")
    
    # For manufactured solution with Dirichlet BC, all should be 0
    boundary_error = (np.max(np.abs(left_vals)) + np.max(np.abs(right_vals)) + 
                     np.max(np.abs(bottom_vals)) + np.max(np.abs(top_vals)))
    print(f"   Boundary values error (should be ~0): {boundary_error:.6e}")
    
    # Step 4: Check boundary condition object
    boundary_condition = create_boundary_condition(config.boundary, boundary_values)
    print(f"\n4. BOUNDARY CONDITION CHECK:")
    print(f"   Type: {type(boundary_condition).__name__}")
    
    # Step 5: Build system matrix
    solver = FiniteVolumeSolverFactory.create_solver(
        mesh, source_function, boundary_condition, config.solver
    )
    
    print(f"\n5. MATRIX ASSEMBLY CHECK:")
    A, b = solver.build_system_matrix()
    
    print(f"   Matrix shape: {A.shape}")
    print(f"   Matrix type: {type(A)}")
    print(f"   Matrix density: {np.count_nonzero(A.toarray())/A.size*100:.1f}%")
    print(f"   Condition number: {np.linalg.cond(A.toarray()):.2e}")
    
    # Check matrix symmetry (should be symmetric for Poisson)
    A_dense = A.toarray()
    is_symmetric = np.allclose(A_dense, A_dense.T, atol=1e-12)
    print(f"   Matrix symmetric: {is_symmetric}")
    
    # Check diagonal dominance
    diag_vals = np.diag(A_dense)
    off_diag_sums = np.sum(np.abs(A_dense), axis=1) - np.abs(diag_vals)
    diag_dominant = np.all(np.abs(diag_vals) >= off_diag_sums)
    print(f"   Diagonally dominant: {diag_dominant}")
    
    # Check RHS
    print(f"   RHS range: [{np.min(b):.6f}, {np.max(b):.6f}]")
    print(f"   RHS sum: {np.sum(b):.6f}")
    
    # Look at specific matrix entries
    center_i, center_j = mesh.nx//2, mesh.ny//2
    center_idx = center_i * mesh.ny + center_j
    print(f"   Center cell ({center_i},{center_j}) -> index {center_idx}")
    print(f"   Matrix row {center_idx}: {A_dense[center_idx, max(0,center_idx-5):center_idx+6]}")
    print(f"   RHS entry {center_idx}: {b[center_idx]:.6f}")
    
    # Step 6: Solve and check
    solution = solver.solve()
    print(f"\n6. SOLUTION CHECK:")
    print(f"   Solution range: [{np.min(solution):.6f}, {np.max(solution):.6f}]")
    print(f"   Solution sum: {np.sum(solution):.6f}")
    print(f"   Solution at center: {solution[center_i, center_j]:.6f}")
    print(f"   Expected at center: {u_exact_mesh[center_i, center_j]:.6f}")
    
    # Compute error
    error = np.abs(solution - u_exact_mesh)
    l2_error = np.sqrt(np.mean(error**2))
    linf_error = np.max(error)
    
    print(f"\n7. ERROR ANALYSIS:")
    print(f"   L2 error: {l2_error:.6e}")
    print(f"   L∞ error: {linf_error:.6e}")
    print(f"   Relative L2 error: {l2_error/np.sqrt(np.mean(u_exact_mesh**2)):.6e}")
    
    # Check if solution is reasonable
    if linf_error > 1e10:
        print("❌ MASSIVE ERROR - Problem in implementation")
        
        # Additional diagnostics
        print(f"\nADDITIONAL DIAGNOSTICS:")
        print(f"   Matrix rank: {np.linalg.matrix_rank(A_dense)}")
        print(f"   Expected rank: {A.shape[0]}")
        print(f"   Matrix null space: {A.shape[0] - np.linalg.matrix_rank(A_dense)}")
        
        # Check if matrix is singular
        try:
            det = np.linalg.det(A_dense)
            print(f"   Determinant: {det:.2e}")
            if abs(det) < 1e-10:
                print("   ⚠️  Matrix is nearly singular!")
        except:
            print("   ❌ Could not compute determinant")
            
        # Check residual
        residual = A_dense @ solution.ravel() - b
        print(f"   Residual norm: {np.linalg.norm(residual):.2e}")
        
    elif linf_error < 1e-2:
        print("✅ Solution looks reasonable")
    else:
        print("⚠️  Solution has some issues but not catastrophic")
    
    return solution, u_exact_mesh, error

def compare_matrix_assembly():
    """Compare matrix assembly between reference and main implementation"""
    print("\n" + "="*60)
    print("MATRIX ASSEMBLY COMPARISON")
    print("="*60)
    
    # Simple 3x3 case for manual verification
    nx, ny = 3, 3
    
    # Reference matrix (manual)
    print("1. REFERENCE MATRIX (3×3 mesh):")
    n_total = nx * ny
    A_ref = np.zeros((n_total, n_total))
    b_ref = np.zeros(n_total)
    
    dx = dy = 1.0/3.0  # For unit domain
    
    def get_idx(i, j):
        return i * ny + j
    
    for i in range(nx):
        for j in range(ny):
            idx = get_idx(i, j)
            
            if i == 0 or i == nx-1 or j == 0 or j == ny-1:
                # Boundary
                A_ref[idx, idx] = 1.0
                b_ref[idx] = 0.0
            else:
                # Interior: -Δu = f
                A_ref[idx, idx] = 2.0/dx**2 + 2.0/dy**2
                A_ref[idx, get_idx(i-1, j)] = -1.0/dx**2
                A_ref[idx, get_idx(i+1, j)] = -1.0/dx**2
                A_ref[idx, get_idx(i, j-1)] = -1.0/dy**2
                A_ref[idx, get_idx(i, j+1)] = -1.0/dy**2
                b_ref[idx] = 2.0  # f = 2
    
    print("Reference matrix A:")
    print(A_ref)
    print("Reference RHS b:")
    print(b_ref)
    
    # Main implementation matrix
    print("\n2. MAIN IMPLEMENTATION MATRIX:")
    from config.config import ProblemConfig
    from src.mesh.mesh_generator import create_mesh
    from src.functions.source_functions import create_source_function, get_boundary_values
    from src.boundary.boundary_conditions import create_boundary_condition
    from src.solver.finite_volume_solver import FiniteVolumeSolverFactory
    
    config = ProblemConfig()
    config.mesh.nx = 3
    config.mesh.ny = 3
    config.function_name = "simple_quadratic"  # f = 2
    config.boundary.bc_type = "dirichlet"
    
    mesh = create_mesh(config.mesh, config.domain)
    source_function = create_source_function(config.function_name)
    boundary_values = get_boundary_values(source_function, mesh, config.boundary)
    boundary_condition = create_boundary_condition(config.boundary, boundary_values)
    
    solver = FiniteVolumeSolverFactory.create_solver(
        mesh, source_function, boundary_condition, config.solver
    )
    
    A_main, b_main = solver.build_system_matrix()
    A_main_dense = A_main.toarray()
    
    print("Main implementation matrix A:")
    print(A_main_dense)
    print("Main implementation RHS b:")
    print(b_main)
    
    # Compare
    print("\n3. COMPARISON:")
    matrix_diff = np.abs(A_ref - A_main_dense)
    rhs_diff = np.abs(b_ref - b_main)
    
    print(f"Max matrix difference: {np.max(matrix_diff):.2e}")
    print(f"Max RHS difference: {np.max(rhs_diff):.2e}")
    
    if np.max(matrix_diff) > 1e-10:
        print("❌ Matrices differ significantly!")
        print("Difference matrix:")
        print(matrix_diff)
    else:
        print("✅ Matrices match")
    
    if np.max(rhs_diff) > 1e-10:
        print("❌ RHS vectors differ!")
        print("RHS difference:", rhs_diff)
    else:
        print("✅ RHS vectors match")

def main():
    """Run complete debugging"""
    print("COMPLETE FINITE VOLUME SOLVER DEBUG")
    print("="*80)
    
    try:
        # 1. Test reference implementation
        u_ref, u_exact_ref, error_ref = minimal_finite_volume_solver()
        
        # 2. Debug main solver
        u_main, u_exact_main, error_main = debug_main_solver()
        
        # 3. Compare matrix assembly
        compare_matrix_assembly()
        
        print("\n" + "="*80)
        print("DEBUGGING COMPLETED")
        print("="*80)
        
        # Final comparison
        if np.max(error_ref) < 1e-2 and np.max(error_main) > 1e10:
            print("❌ Main implementation has serious bugs - matrix assembly issue")
        elif np.max(error_ref) > 1e-2:
            print("❌ Even reference implementation fails - conceptual issue")
        else:
            print("✅ Both implementations work correctly")
            
    except Exception as e:
        print(f"Error during debugging: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()