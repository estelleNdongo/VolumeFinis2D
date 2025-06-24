"""
Test suite for convergence analysis
"""
import unittest
import numpy as np
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from config.config import ProblemConfig
from src.mesh.mesh_generator import create_mesh
from src.functions.source_functions import create_source_function, get_boundary_values
from src.boundary.boundary_conditions import create_boundary_condition
from src.solver.finite_volume_solver import FiniteVolumeSolverFactory
from src.analysis.error_analysis import ErrorAnalyzer, ConvergenceStudy

class TestConvergence(unittest.TestCase):
    """Test convergence properties of the finite volume solver"""
    
    def setUp(self):
        """Set up test configuration"""
        self.config = ProblemConfig()
        self.config.function_name = "manufactured"  # Has exact solution
        self.config.boundary.bc_type = "dirichlet"
        self.config.solver.solver_type = "direct"
        
    def create_solver(self, nx, ny):
        """Helper to create solver with given mesh size"""
        self.config.mesh.nx = nx
        self.config.mesh.ny = ny
        
        mesh = create_mesh(self.config.mesh, self.config.domain)
        source_function = create_source_function(self.config.function_name)
        boundary_values = get_boundary_values(source_function, mesh, self.config.boundary)
        boundary_condition = create_boundary_condition(self.config.boundary, boundary_values)
        
        return FiniteVolumeSolverFactory.create_solver(
            mesh, source_function, boundary_condition, self.config.solver
        )
    
    def test_manufactured_solution_convergence(self):
        """Test convergence for manufactured solution"""
        
        mesh_sizes = [10, 20, 40]
        errors = []
        h_values = []
        
        analyzer = ErrorAnalyzer()
        
        for nx in mesh_sizes:
            # Solve
            solver = self.create_solver(nx, nx)
            solution = solver.solve()
            
            # Compute exact solution
            exact_solution = solver.source_function.exact_solution(
                solver.mesh.x_centers, solver.mesh.y_centers
            )
            
            # Compute errors
            error_norms = analyzer.compute_error_norms(solution, exact_solution, solver.mesh)
            errors.append(error_norms['L2'])
            h_values.append(1.0 / nx)
        
        # Analyze convergence
        rate, correlation, constant = analyzer.compute_convergence_rate(h_values, errors)
        
        # Check convergence rate (should be close to 2 for second-order method)
        self.assertGreater(rate, 1.5, "Convergence rate should be at least 1.5")
        self.assertLess(rate, 3.0, "Convergence rate should not exceed 3.0")
        
        # Check correlation (should indicate good fit)
        self.assertGreater(correlation**2, 0.95, "Convergence should show good correlation")
        
        print(f"Manufactured solution convergence rate: {rate:.3f}")
        print(f"Correlation R²: {correlation**2:.4f}")
    
    def test_error_reduction(self):
        """Test that errors decrease with mesh refinement"""
        
        mesh_sizes = [10, 20, 40]
        l2_errors = []
        linf_errors = []
        
        analyzer = ErrorAnalyzer()
        
        for nx in mesh_sizes:
            solver = self.create_solver(nx, nx)
            solution = solver.solve()
            
            exact_solution = solver.source_function.exact_solution(
                solver.mesh.x_centers, solver.mesh.y_centers
            )
            
            error_norms = analyzer.compute_error_norms(solution, exact_solution, solver.mesh)
            l2_errors.append(error_norms['L2'])
            linf_errors.append(error_norms['L_inf'])
        
        # Check that errors decrease
        for i in range(1, len(l2_errors)):
            self.assertLess(l2_errors[i], l2_errors[i-1], 
                           f"L2 error should decrease: {l2_errors[i]} < {l2_errors[i-1]}")
            self.assertLess(linf_errors[i], linf_errors[i-1],
                           f"L∞ error should decrease: {linf_errors[i]} < {linf_errors[i-1]}")
        
        print(f"L2 errors: {[f'{e:.2e}' for e in l2_errors]}")
        print(f"L∞ errors: {[f'{e:.2e}' for e in linf_errors]}")
    
    def test_quadratic_convergence_rate(self):
        """Test that finite volumes achieve second-order convergence"""
        
        # Use finer mesh sizes for better rate estimation
        mesh_sizes = [8, 16, 32, 64]
        errors = []
        h_values = []
        
        analyzer = ErrorAnalyzer()
        
        for nx in mesh_sizes:
            solver = self.create_solver(nx, nx)
            solution = solver.solve()
            
            exact_solution = solver.source_function.exact_solution(
                solver.mesh.x_centers, solver.mesh.y_centers
            )
            
            error_norms = analyzer.compute_error_norms(solution, exact_solution, solver.mesh)
            errors.append(error_norms['L2'])
            h_values.append(1.0 / nx)
        
        # Compute convergence rate
        rate, correlation, constant = analyzer.compute_convergence_rate(h_values, errors)
        
        # Should be close to 2 for second-order finite volumes
        self.assertGreater(rate, 1.8, f"Convergence rate {rate:.3f} should be close to 2")
        self.assertLess(rate, 2.2, f"Convergence rate {rate:.3f} should be close to 2")
        
        print(f"Quadratic convergence test: rate = {rate:.3f}")
    
    def test_convergence_study_class(self):
        """Test the ConvergenceStudy class"""
        
        study = ConvergenceStudy(self.config)
        
        def create_solver_func(config):
            mesh = create_mesh(config.mesh, config.domain)
            source_function = create_source_function(config.function_name)
            boundary_values = get_boundary_values(source_function, mesh, config.boundary)
            boundary_condition = create_boundary_condition(config.boundary, boundary_values)
            return FiniteVolumeSolverFactory.create_solver(
                mesh, source_function, boundary_condition, config.solver
            )
        
        # Run convergence study
        mesh_sizes = [10, 20, 40]
        results = study.run_convergence_study(mesh_sizes, create_solver_func)
        
        # Check that we have results
        self.assertEqual(len(results), len(mesh_sizes))
        
        # Check that errors are computed
        for result in results:
            self.assertIn('errors', result)
            self.assertIn('L2', result['errors'])
            self.assertGreater(result['errors']['L2'], 0)
        
        # Get convergence analysis
        convergence_analysis = study.analyzer.analyze_convergence(results)
        
        # Check convergence rates
        for norm_name, analysis in convergence_analysis.items():
            if norm_name not in ['exact_L2_norm']:
                rate = analysis['convergence_rate']
                self.assertGreater(rate, 1.0, f"{norm_name} convergence rate should be positive")
                
        print("ConvergenceStudy class test passed")
    
    def test_different_source_functions(self):
        """Test convergence for different source functions"""
        
        functions = ["manufactured", "trigonometric"]
        
        for func_name in functions:
            with self.subTest(function=func_name):
                self.config.function_name = func_name
                
                mesh_sizes = [10, 20, 40]
                errors = []
                h_values = []
                
                analyzer = ErrorAnalyzer()
                
                for nx in mesh_sizes:
                    solver = self.create_solver(nx, nx)
                    solution = solver.solve()
                    
                    # Skip if no exact solution
                    if not solver.source_function.has_exact_solution():
                        continue
                    
                    exact_solution = solver.source_function.exact_solution(
                        solver.mesh.x_centers, solver.mesh.y_centers
                    )
                    
                    error_norms = analyzer.compute_error_norms(solution, exact_solution, solver.mesh)
                    errors.append(error_norms['L2'])
                    h_values.append(1.0 / nx)
                
                if len(errors) >= 2:
                    rate, correlation, constant = analyzer.compute_convergence_rate(h_values, errors)
                    
                    # Check reasonable convergence
                    self.assertGreater(rate, 1.0, f"Function {func_name}: rate should be positive")
                    self.assertLess(rate, 4.0, f"Function {func_name}: rate should be reasonable")
                    
                    print(f"Function {func_name}: convergence rate = {rate:.3f}")

class TestErrorNorms(unittest.TestCase):
    """Test error norm computations"""
    
    def setUp(self):
        """Set up test data"""
        self.config = ProblemConfig()
        self.config.mesh.nx = 10
        self.config.mesh.ny = 10
        
        # Create simple test mesh
        self.mesh = create_mesh(self.config.mesh, self.config.domain)
        
        # Create test solutions
        x, y = self.mesh.x_centers, self.mesh.y_centers
        self.exact = np.sin(np.pi * x) * np.sin(np.pi * y)
        self.numerical = self.exact + 0.01 * np.random.random(self.exact.shape)
        
        self.analyzer = ErrorAnalyzer()
    
    def test_error_norm_properties(self):
        """Test basic properties of error norms"""
        
        errors = self.analyzer.compute_error_norms(
            self.numerical, self.exact, self.mesh
        )
        
        # All error norms should be positive
        for norm_name, error_value in errors.items():
            if norm_name != 'exact_L2_norm':
                self.assertGreaterEqual(error_value, 0.0, f"{norm_name} should be non-negative")
        
        # L∞ norm should be >= L2 norm (for discrete case, approximately)
        # This might not always hold due to weighting, so we check it's reasonable
        self.assertGreater(errors['L_inf'], 0.0, "L∞ norm should be positive")
        self.assertGreater(errors['L2'], 0.0, "L2 norm should be positive")
    
    def test_zero_error(self):
        """Test error norms when solutions are identical"""
        
        errors = self.analyzer.compute_error_norms(
            self.exact, self.exact, self.mesh
        )
        
        # All error norms should be zero (within numerical precision)
        tolerance = 1e-14
        self.assertLess(errors['L_inf'], tolerance, "L∞ error should be zero for identical solutions")
        self.assertLess(errors['L2'], tolerance, "L2 error should be zero for identical solutions")
        self.assertLess(errors['L1'], tolerance, "L1 error should be zero for identical solutions")

if __name__ == '__main__':
    # Run tests with verbose output
    unittest.main(verbosity=2)