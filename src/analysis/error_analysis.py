"""
Error analysis and convergence study module
"""
import numpy as np
from typing import List, Dict, Tuple
import matplotlib.pyplot as plt
from scipy import stats

class ErrorAnalyzer:
    """Class for computing various error norms and convergence rates"""
    
    def __init__(self):
        self.error_history = []
        self.mesh_sizes = []
        self.convergence_rates = {}
    
    def compute_error_norms(self, numerical_solution: np.ndarray, 
                           exact_solution: np.ndarray, 
                           mesh) -> Dict[str, float]:
        """
        Compute various error norms between numerical and exact solutions
        
        Args:
            numerical_solution: Numerical solution on mesh
            exact_solution: Exact solution on same mesh
            mesh: Mesh object with geometric information
            
        Returns:
            Dictionary with error norms
        """
        
        # Ensure arrays have the same shape
        if numerical_solution.shape != exact_solution.shape:
            raise ValueError("Numerical and exact solutions must have same shape")
        
        # Compute pointwise error
        error = numerical_solution - exact_solution
        
        # L∞ norm (maximum norm)
        l_inf_error = np.max(np.abs(error))
        
        # L2 norm (weighted by cell volumes)
        l2_error_squared = 0.0
        exact_l2_norm_squared = 0.0
        
        for i in range(mesh.nx):
            for j in range(mesh.ny):
                volume = mesh.get_cell_volume(i, j)
                l2_error_squared += error[i, j]**2 * volume
                exact_l2_norm_squared += exact_solution[i, j]**2 * volume
        
        l2_error = np.sqrt(l2_error_squared)
        exact_l2_norm = np.sqrt(exact_l2_norm_squared)
        
        # Relative L2 norm
        relative_l2_error = l2_error / exact_l2_norm if exact_l2_norm > 0 else l2_error
        
        # L1 norm
        l1_error = 0.0
        for i in range(mesh.nx):
            for j in range(mesh.ny):
                volume = mesh.get_cell_volume(i, j)
                l1_error += np.abs(error[i, j]) * volume
        
        # RMS error
        total_volume = (mesh.x_max - mesh.x_min) * (mesh.y_max - mesh.y_min)
        rms_error = np.sqrt(l2_error_squared / total_volume)
        
        # Mean absolute error
        mean_abs_error = l1_error / total_volume
        
        return {
            'L_inf': l_inf_error,
            'L2': l2_error,
            'L2_relative': relative_l2_error,
            'L1': l1_error,
            'RMS': rms_error,
            'MAE': mean_abs_error,
            'exact_L2_norm': exact_l2_norm
        }
    
    def compute_convergence_rate(self, mesh_sizes: List[float], 
                                errors: List[float], 
                                method: str = 'least_squares') -> Tuple[float, float, float]:
        """
        Compute convergence rate from a series of mesh refinements
        
        Args:
            mesh_sizes: List of characteristic mesh sizes (h)
            errors: List of corresponding errors
            method: Method for computing rate ('least_squares' or 'successive')
            
        Returns:
            Tuple of (convergence_rate, correlation_coefficient, constant)
        """
        
        if len(mesh_sizes) != len(errors):
            raise ValueError("mesh_sizes and errors must have same length")
        
        if len(mesh_sizes) < 2:
            raise ValueError("Need at least 2 data points for convergence analysis")
        
        mesh_sizes = np.array(mesh_sizes)
        errors = np.array(errors)
        
        # Remove zero errors to avoid log(0)
        valid_mask = errors > 0
        mesh_sizes = mesh_sizes[valid_mask]
        errors = errors[valid_mask]
        
        if len(errors) < 2:
            return 0.0, 0.0, 0.0
        
        if method == 'least_squares':
            # Fit log(error) = log(C) + p*log(h)
            # where p is the convergence rate
            log_h = np.log(mesh_sizes)
            log_error = np.log(errors)
            
            # Linear regression
            slope, intercept, r_value, p_value, std_err = stats.linregress(log_h, log_error)
            
            convergence_rate = slope
            correlation_coeff = r_value
            constant = np.exp(intercept)
            
        elif method == 'successive':
            # Compute rate between successive refinements
            rates = []
            for i in range(1, len(errors)):
                if errors[i] > 0 and errors[i-1] > 0:
                    rate = np.log(errors[i] / errors[i-1]) / np.log(mesh_sizes[i] / mesh_sizes[i-1])
                    rates.append(rate)
            
            convergence_rate = np.mean(rates) if rates else 0.0
            correlation_coeff = 1.0  # Not applicable for successive method
            constant = errors[0] / (mesh_sizes[0] ** convergence_rate) if convergence_rate != 0 else errors[0]
            
        else:
            raise ValueError(f"Unknown method: {method}")
        
        return convergence_rate, correlation_coeff, constant
    
    def analyze_convergence(self, solver_results: List[Dict]) -> Dict[str, Dict]:
        """
        Analyze convergence from multiple solver results
        
        Args:
            solver_results: List of dictionaries containing:
                - 'mesh_size': characteristic mesh size
                - 'errors': dictionary of error norms
                - 'solve_time': computation time
                - 'mesh': mesh object
                
        Returns:
            Dictionary with convergence analysis for each error norm
        """
        
        # Extract data
        mesh_sizes = []
        error_data = {}
        solve_times = []
        
        for result in solver_results:
            mesh_sizes.append(result['mesh_size'])
            solve_times.append(result['solve_time'])
            
            for norm_name, error_value in result['errors'].items():
                if norm_name not in error_data:
                    error_data[norm_name] = []
                error_data[norm_name].append(error_value)
        
        # Analyze convergence for each error norm
        convergence_analysis = {}
        
        for norm_name, errors in error_data.items():
            if norm_name in ['exact_L2_norm']:  # Skip non-error quantities
                continue
                
            try:
                rate, correlation, constant = self.compute_convergence_rate(mesh_sizes, errors)
                
                # Theoretical convergence rate for finite volumes (2nd order)
                theoretical_rate = 2.0
                
                convergence_analysis[norm_name] = {
                    'errors': errors,
                    'mesh_sizes': mesh_sizes,
                    'convergence_rate': rate,
                    'correlation': correlation,
                    'constant': constant,
                    'theoretical_rate': theoretical_rate,
                    'rate_ratio': rate / theoretical_rate if theoretical_rate != 0 else 0,
                    'solve_times': solve_times
                }
            except Exception as e:
                print(f"Warning: Could not compute convergence for {norm_name}: {e}")
                convergence_analysis[norm_name] = {
                    'errors': errors,
                    'mesh_sizes': mesh_sizes,
                    'convergence_rate': 0.0,
                    'correlation': 0.0,
                    'constant': 0.0,
                    'theoretical_rate': 2.0,
                    'rate_ratio': 0.0,
                    'solve_times': solve_times
                }
        
        return convergence_analysis
    
    def estimate_discretization_error(self, coarse_solution: np.ndarray,
                                     fine_solution: np.ndarray,
                                     refinement_ratio: int = 2) -> float:
        """
        Estimate discretization error using Richardson extrapolation
        
        Args:
            coarse_solution: Solution on coarse mesh
            fine_solution: Solution on fine mesh (interpolated to coarse mesh)
            refinement_ratio: Mesh refinement ratio
            
        Returns:
            Estimated discretization error
        """
        
        if coarse_solution.shape != fine_solution.shape:
            raise ValueError("Solutions must have same shape")
        
        # Assuming 2nd order accuracy
        p = 2.0
        r = refinement_ratio
        
        # Richardson extrapolation error estimate
        error_estimate = np.abs(fine_solution - coarse_solution) / (r**p - 1)
        
        return np.max(error_estimate)
    
    def compute_grid_convergence_index(self, solutions: List[np.ndarray],
                                      mesh_sizes: List[float],
                                      safety_factor: float = 1.25) -> Dict[str, float]:
        """
        Compute Grid Convergence Index (GCI) for uncertainty quantification
        
        Args:
            solutions: List of solutions on different meshes
            mesh_sizes: Corresponding mesh sizes
            safety_factor: Safety factor (typically 1.25 for 2D)
            
        Returns:
            Dictionary with GCI values
        """
        
        if len(solutions) < 2:
            raise ValueError("Need at least 2 solutions")
        
        gci_results = {}
        
        for i in range(len(solutions) - 1):
            # Coarse and fine solutions
            coarse = solutions[i]
            fine = solutions[i + 1]
            
            h_coarse = mesh_sizes[i]
            h_fine = mesh_sizes[i + 1]
            
            r = h_coarse / h_fine  # Refinement ratio
            
            # Compute apparent order of accuracy
            if i < len(solutions) - 2:
                # Use three grids to estimate order
                medium = solutions[i + 1]
                finest = solutions[i + 2]
                
                h_medium = mesh_sizes[i + 1]
                h_finest = mesh_sizes[i + 2]
                
                # Estimate order using three solutions
                r21 = h_medium / h_finest
                r32 = h_coarse / h_medium
                
                epsilon32 = np.mean(np.abs(coarse - medium))
                epsilon21 = np.mean(np.abs(medium - finest))
                
                if epsilon21 > 0 and epsilon32 > 0:
                    p = np.log(epsilon32 / epsilon21) / np.log(r32 / r21)
                else:
                    p = 2.0  # Assumed order
            else:
                p = 2.0  # Assumed order for finite volumes
            
            # Ensure solutions have same shape (interpolate if necessary)
            if coarse.shape != fine.shape:
                # Simple average for different grid sizes
                epsilon = np.mean(np.abs(coarse.ravel()[:min(coarse.size, fine.size)] - 
                                       fine.ravel()[:min(coarse.size, fine.size)]))
            else:
                epsilon = np.mean(np.abs(coarse - fine))
            
            # GCI calculation
            gci = safety_factor * epsilon / (r**p - 1)
            
            gci_results[f'GCI_{i+1}_{i+2}'] = {
                'gci': gci,
                'apparent_order': p,
                'refinement_ratio': r,
                'relative_error': epsilon / np.mean(np.abs(fine))
            }
        
        return gci_results

class ConvergenceStudy:
    """Class for performing systematic convergence studies"""
    
    def __init__(self, base_config):
        self.base_config = base_config
        self.results = []
        self.analyzer = ErrorAnalyzer()
    
    def run_convergence_study(self, mesh_sizes: List[int], 
                            create_solver_func,
                            exact_solution_func = None) -> List[Dict]:
        """
        Run convergence study for multiple mesh sizes
        
        Args:
            mesh_sizes: List of mesh sizes (nx = ny)
            create_solver_func: Function to create solver given mesh size
            exact_solution_func: Function to compute exact solution
            
        Returns:
            List of results for each mesh size
        """
        
        print("Starting convergence study...")
        print(f"Mesh sizes: {mesh_sizes}")
        
        study_results = []
        
        for nx in mesh_sizes:
            print(f"\n--- Solving for mesh size {nx}x{nx} ---")
            
            # Update mesh configuration
            config = self.base_config
            config.mesh.nx = nx
            config.mesh.ny = nx
            
            # Create solver
            solver = create_solver_func(config)
            
            # Solve
            solution = solver.solve()
            
            # Compute characteristic mesh size
            h = 1.0 / nx  # For unit square
            
            # Compute errors if exact solution is available
            errors = {}
            if exact_solution_func and solver.source_function.has_exact_solution():
                # Evaluate exact solution on mesh
                exact_solution = solver.source_function.exact_solution(
                    solver.mesh.x_centers, solver.mesh.y_centers
                )
                
                # Compute error norms
                errors = self.analyzer.compute_error_norms(
                    solution, exact_solution, solver.mesh
                )
                
                print(f"L∞ error: {errors['L_inf']:.2e}")
                print(f"L2 error: {errors['L2']:.2e}")
                print(f"Relative L2 error: {errors['L2_relative']:.2e}")
            
            # Store results
            result = {
                'mesh_size': h,
                'nx': nx,
                'ny': nx,
                'solution': solution,
                'errors': errors,
                'solve_time': solver.solve_time,
                'mesh': solver.mesh,
                'solver_info': solver.get_solver_info()
            }
            
            study_results.append(result)
        
        # Analyze convergence
        if study_results and study_results[0]['errors']:
            print("\n--- Convergence Analysis ---")
            convergence_analysis = self.analyzer.analyze_convergence(study_results)
            
            for norm_name, analysis in convergence_analysis.items():
                rate = analysis['convergence_rate']
                correlation = analysis['correlation']
                print(f"{norm_name:12s}: rate = {rate:6.3f}, R² = {correlation**2:6.4f}")
        
        self.results = study_results
        return study_results
    
    def get_convergence_summary(self) -> Dict:
        """Get summary of convergence study results"""
        if not self.results:
            return {}
        
        # Extract key metrics
        mesh_sizes = [r['mesh_size'] for r in self.results]
        solve_times = [r['solve_time'] for r in self.results]
        
        summary = {
            'mesh_sizes': mesh_sizes,
            'solve_times': solve_times,
            'total_study_time': sum(solve_times),
            'efficiency': len(mesh_sizes) / sum(solve_times) if sum(solve_times) > 0 else 0
        }
        
        # Add error information if available
        if self.results[0]['errors']:
            convergence_analysis = self.analyzer.analyze_convergence(self.results)
            summary['convergence_analysis'] = convergence_analysis
        
        return summary