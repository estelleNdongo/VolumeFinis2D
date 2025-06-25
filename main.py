"""
Main script for finite volume Poisson solver
"""
import argparse
import numpy as np
import sys
import os



# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from config.config import ProblemConfig, PREDEFINED_CONFIGS
from src.mesh.mesh_generator import create_mesh
from src.functions.source_functions import create_source_function, get_boundary_values
from src.boundary.boundary_conditions import create_boundary_condition
from src.solver.finite_volume_solver import FiniteVolumeSolverFactory
from src.analysis.error_analysis import ConvergenceStudy, ErrorAnalyzer
from src.analysis.visualization import create_visualizer

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='2D Finite Volume Poisson Solver')
    
    # Mesh parameters
    parser.add_argument('--nx', type=int, default=20, help='Number of cells in x-direction')
    parser.add_argument('--ny', type=int, default=20, help='Number of cells in y-direction')
    parser.add_argument('--mesh-type', type=str, default='uniform', 
                       choices=['uniform', 'refined', 'stretched'],
                       help='Type of mesh')
    
    # Problem parameters
    parser.add_argument('--function', type=str, default='quadratic',
                       choices=['quadratic', 'manufactured', 'trigonometric', 'polynomial', 'gaussian'],
                       help='Source function type')
    parser.add_argument('--bc-type', type=str, default='dirichlet',
                       choices=['dirichlet', 'neumann', 'mixed', 'robin'],
                       help='Boundary condition type')
    
    # Solver parameters
    parser.add_argument('--solver', type=str, default='direct',
                       choices=['direct', 'cg', 'gmres', 'bicgstab'],
                       help='Solver type')
    parser.add_argument('--tolerance', type=float, default=1e-10,
                       help='Solver tolerance')
    
    # Analysis options
    parser.add_argument('--convergence-study', action='store_true',
                       help='Perform convergence study')
    parser.add_argument('--mesh-sizes', nargs='+', type=int, default=[10, 20, 40, 80],
                       help='Mesh sizes for convergence study')
    
    # Visualization options
    parser.add_argument('--plot', action='store_true', default=True,
                       help='Generate plots')
    parser.add_argument('--no-plot', action='store_false', dest='plot',
                       help='Disable plotting')
    parser.add_argument('--save-plots', action='store_true', default=True,
                       help='Save plots to files')
    
    # Configuration options
    parser.add_argument('--config', type=str, choices=list(PREDEFINED_CONFIGS.keys()),
                       help='Use predefined configuration')
    
    # Output options
    parser.add_argument('--output-dir', type=str, default='results',
                       help='Output directory for results')
    parser.add_argument('--verbose', action='store_true',
                       help='Verbose output')
    
    return parser.parse_args()

def create_solver_from_config(config):
    """Create solver from configuration"""
    
    # Create mesh
    print(f"Creating {config.mesh.mesh_type} mesh: {config.mesh.nx}×{config.mesh.ny}")
    mesh = create_mesh(config.mesh, config.domain)
    
    # Create source function
    print(f"Creating source function: {config.function_name}")
    source_function = create_source_function(config.function_name)
    
    # Get boundary values
    boundary_values = get_boundary_values(source_function, mesh, config.boundary)
    
    # Create boundary condition
    print(f"Creating boundary conditions: {config.boundary.bc_type}")
    boundary_condition = create_boundary_condition(config.boundary, boundary_values)
    
    # Create solver
    print(f"Creating solver: {config.solver.solver_type}")
    solver = FiniteVolumeSolverFactory.create_solver(
        mesh, source_function, boundary_condition, config.solver
    )
    
    return solver

def solve_single_case(config, visualizer=None):
    """Solve single case and optionally visualize results"""
    
    print("\n" + "="*60)
    print("SOLVING SINGLE CASE")
    print("="*60)
    
    # Create solver
    solver = create_solver_from_config(config)
    
    # Solve
    print("\nSolving system...")
    solution = solver.solve()
    
    # Print solver information
    solver_info = solver.get_solver_info()
    print(f"\nSolver completed successfully!")
    print(f"Solve time: {solver_info['solve_time']:.4f} seconds")
    print(f"Mesh size: {solver_info['mesh_size'][0]}×{solver_info['mesh_size'][1]}")
    print(f"Total cells: {solver_info['total_cells']}")
    
    # Compute errors if exact solution is available
    errors = {}
    if solver.source_function.has_exact_solution():
        print("\nComputing errors...")
        exact_solution = solver.source_function.exact_solution(
            solver.mesh.x_centers, solver.mesh.y_centers
        )
        
        analyzer = ErrorAnalyzer()
        errors = analyzer.compute_error_norms(solution, exact_solution, solver.mesh)
        
        print("\nError Analysis:")
        print(f"L∞ error:        {errors['L_inf']:.6e}")
        print(f"L2 error:        {errors['L2']:.6e}")
        print(f"Relative L2:     {errors['L2_relative']:.6e}")
        print(f"RMS error:       {errors['RMS']:.6e}")
        print(f"Mean abs error:  {errors['MAE']:.6e}")
        
        # Visualize if requested
        if visualizer:
            print("\nGenerating plots...")
            
            # Solution comparison
            visualizer.plot_solution_interactive(solution, solver.mesh)
           
            
            # Error distribution
            visualizer.plot_error_distribution_interactive(solution, exact_solution, solver.mesh)
            
            # 3D plots
            visualizer.plot_solution_3d(
                solution, solver.mesh,
                title="Numerical Solution (3D)",
                filename="solution_3d"
            )
            
            visualizer.plot_solution_3d(
                exact_solution, solver.mesh,
                title="Exact Solution (3D)",
                filename="exact_solution_3d"
            )
    
    else:
        print("\nNo exact solution available for error analysis")
        
        # Visualize solution only
        if visualizer:
            print("\nGenerating plots...")
            visualizer.plot_solution_2d(
                solution, solver.mesh,
                title="Numerical Solution",
                filename="solution_2d"
            )
            
            visualizer.plot_solution_3d(
                solution, solver.mesh,
                title="Numerical Solution (3D)",
                filename="solution_3d"
            )
    
    # Plot mesh
    if visualizer:
        visualizer.plot_mesh(
            solver.mesh,
            title="Computational Mesh",
            filename="mesh"
        )
    
    return {
        'solution': solution,
        'errors': errors,
        'solver_info': solver_info,
        'mesh': solver.mesh
    }

def run_convergence_study(config, mesh_sizes, visualizer=None):
    """Run convergence study"""
    
    print("\n" + "="*60)
    print("CONVERGENCE STUDY")
    print("="*60)
    
    # Create convergence study
    study = ConvergenceStudy(config)
    
    # Define solver creation function
    def create_solver_func(study_config):
        return create_solver_from_config(study_config)
    
    # Run study
    results = study.run_convergence_study(
        mesh_sizes, 
        create_solver_func,
        exact_solution_func=lambda x, y: config.source_function.exact_solution(x, y) if hasattr(config, 'source_function') else None
    )
    
    # Get convergence analysis
    if results and results[0]['errors']:
        analyzer = ErrorAnalyzer()
        convergence_analysis = analyzer.analyze_convergence(results)
        
        print("\n" + "="*60)
        print("CONVERGENCE ANALYSIS SUMMARY")
        print("="*60)
        
        for norm_name, analysis in convergence_analysis.items():
            if norm_name in ['exact_L2_norm']:
                continue
            
            rate = analysis['convergence_rate']
            correlation = analysis['correlation']
            theoretical_rate = analysis['theoretical_rate']
            rate_ratio = analysis['rate_ratio']
            
            print(f"\n{norm_name} Error:")
            print(f"  Observed rate:     {rate:6.3f}")
            print(f"  Theoretical rate:  {theoretical_rate:6.3f}")
            print(f"  Rate ratio:        {rate_ratio:6.3f}")
            print(f"  Correlation (R²):  {correlation**2:6.4f}")
            
            # Print error values
            print(f"  Errors: {[f'{e:.2e}' for e in analysis['errors']]}")
        
        # Visualize convergence
        if visualizer:
            print("\nGenerating convergence plots...")
            visualizer.plot_convergence_study(
                convergence_analysis,
                title="Convergence Study",
                filename="convergence_study"
            )
        
        return convergence_analysis
    else:
        print("No error analysis available for convergence study")
        return None

def main():
    """Main function"""
    
    print("2D Finite Volume Poisson Solver")
    print("================================")
    
    # Parse arguments
    args = parse_arguments()
    
    # Create configuration
    if args.config:
        print(f"Using predefined configuration: {args.config}")
        config = ProblemConfig()
        config.update_from_dict(PREDEFINED_CONFIGS[args.config])
    else:
        config = ProblemConfig.from_args(args)
    
    # Update configuration from command line arguments
    config.mesh.nx = args.nx
    config.mesh.ny = args.ny
    config.mesh.mesh_type = args.mesh_type
    config.function_name = args.function
    config.boundary.bc_type = args.bc_type
    config.solver.solver_type = args.solver
    config.solver.tolerance = args.tolerance
    config.visualization.save_plots = args.save_plots
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create visualizer
    visualizer = None
    if args.plot:
        visualizer = create_visualizer(config)
    
    # Print configuration summary
    if args.verbose:
        print("\nConfiguration Summary:")
        print(f"  Domain: [{config.domain.x_min}, {config.domain.x_max}] × [{config.domain.y_min}, {config.domain.y_max}]")
        print(f"  Mesh: {config.mesh.nx}×{config.mesh.ny} ({config.mesh.mesh_type})")
        print(f"  Function: {config.function_name}")
        print(f"  Boundary: {config.boundary.bc_type}")
        print(f"  Solver: {config.solver.solver_type}")
    
    try:
        if args.convergence_study:
            # Run convergence study
            convergence_results = run_convergence_study(
                config, args.mesh_sizes, visualizer
            )
        else:
            # Solve single case
            single_results = solve_single_case(config, visualizer)
        
        print("\n" + "="*60)
        print("COMPUTATION COMPLETED SUCCESSFULLY")
        print("="*60)
        
        if args.save_plots and visualizer:
            print(f"Plots saved to: plots/")
        
    except Exception as e:
        print(f"\nError during computation: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)