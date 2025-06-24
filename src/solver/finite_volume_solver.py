"""
Finite Volume Solver for 2D Poisson equation
"""
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from typing import Tuple, Optional
import time

class FiniteVolumeSolver:
    """Finite volume solver for -Δu = f"""
    
    def __init__(self, mesh, source_function, boundary_condition, solver_config):
        self.mesh = mesh
        self.source_function = source_function
        self.boundary_condition = boundary_condition
        self.solver_config = solver_config
        
        # Solution storage
        self.solution = None
        self.residual_history = []
        self.solve_time = 0.0
        
    def build_system_matrix(self) -> Tuple[sp.csr_matrix, np.ndarray]:
        """Build the system matrix A and right-hand side vector b"""
        
        nx, ny = self.mesh.nx, self.mesh.ny
        n_cells = nx * ny
        
        # Initialize sparse matrix storage
        row_indices = []
        col_indices = []
        data = []
        
        # Right-hand side vector
        b = np.zeros(n_cells)
        
        def get_global_index(i: int, j: int) -> int:
            """Convert (i,j) cell indices to global matrix index"""
            return i * ny + j
        
        # Assemble interior equations
        for i in range(nx):
            for j in range(ny):
                idx = get_global_index(i, j)
                
                # Get cell volume and coordinates
                volume = self.mesh.get_cell_volume(i, j)
                x_center = self.mesh.x_centers[i, j]
                y_center = self.mesh.y_centers[i, j]
                
                # Source term contribution
                f_value = self.source_function.evaluate(x_center, y_center)
                b[idx] = f_value * volume
                
                # Initialize diagonal term
                diagonal_coeff = 0.0
                
                # East face (i+1/2, j)
                if i < nx - 1:
                    neighbor_idx = get_global_index(i + 1, j)
                    face_area = self.mesh.get_face_area('east', i, j)
                    distance = self.mesh.get_face_distance('east', i, j)
                    coeff = face_area / distance
                    
                    # Add off-diagonal term
                    row_indices.append(idx)
                    col_indices.append(neighbor_idx)
                    data.append(-coeff)
                    
                    # Update diagonal
                    diagonal_coeff += coeff
                
                # West face (i-1/2, j)
                if i > 0:
                    neighbor_idx = get_global_index(i - 1, j)
                    face_area = self.mesh.get_face_area('west', i, j)
                    distance = self.mesh.get_face_distance('west', i, j)
                    coeff = face_area / distance
                    
                    # Add off-diagonal term
                    row_indices.append(idx)
                    col_indices.append(neighbor_idx)
                    data.append(-coeff)
                    
                    # Update diagonal
                    diagonal_coeff += coeff
                
                # North face (i, j+1/2)
                if j < ny - 1:
                    neighbor_idx = get_global_index(i, j + 1)
                    face_area = self.mesh.get_face_area('north', i, j)
                    distance = self.mesh.get_face_distance('north', i, j)
                    coeff = face_area / distance
                    
                    # Add off-diagonal term
                    row_indices.append(idx)
                    col_indices.append(neighbor_idx)
                    data.append(-coeff)
                    
                    # Update diagonal
                    diagonal_coeff += coeff
                
                # South face (i, j-1/2)
                if j > 0:
                    neighbor_idx = get_global_index(i, j - 1)
                    face_area = self.mesh.get_face_area('south', i, j)
                    distance = self.mesh.get_face_distance('south', i, j)
                    coeff = face_area / distance
                    
                    # Add off-diagonal term
                    row_indices.append(idx)
                    col_indices.append(neighbor_idx)
                    data.append(-coeff)
                    
                    # Update diagonal
                    diagonal_coeff += coeff
                
                # Add diagonal term
                row_indices.append(idx)
                col_indices.append(idx)
                data.append(diagonal_coeff)
        
        # Create sparse matrix
        A = sp.csr_matrix((data, (row_indices, col_indices)), shape=(n_cells, n_cells))
        
        # Apply boundary conditions
        boundary_info = {}
        A_dense = A.toarray()  # Convert to dense BEFORE applying BC
        self.boundary_condition.apply_to_matrix(A_dense, b, self.mesh, boundary_info)
      

        # Convert back to sparse format
        A = sp.csr_matrix(A_dense)
        print(f"DEBUG: First boundary row after BC: {A_dense[0, :5]}")
        print(f"DEBUG: Matrix rank after BC: {np.linalg.matrix_rank(A_dense)}")
        
        return A, b
    
    def solve_direct(self, A: sp.csr_matrix, b: np.ndarray) -> np.ndarray:
        """Solve system using direct method"""
        start_time = time.time()
        
        try:
            # Try sparse direct solver first
            solution = spla.spsolve(A, b)
        except Exception as e:
            print(f"Sparse solver failed: {e}")
            # Fallback to dense solver
            solution = np.linalg.solve(A.toarray(), b)
        
        self.solve_time = time.time() - start_time
        return solution
    
    def solve_iterative(self, A: sp.csr_matrix, b: np.ndarray) -> np.ndarray:
        """Solve system using iterative method"""
        
        def callback(residual):
            self.residual_history.append(residual)
        
        start_time = time.time()
        
        # Choose iterative solver based on configuration
        solver_type = self.solver_config.solver_type.lower()
        tolerance = self.solver_config.tolerance
        max_iter = self.solver_config.max_iterations
        
        if solver_type == "cg":
            solution, info = spla.cg(A, b, tol=tolerance, maxiter=max_iter, callback=callback)
        elif solver_type == "gmres":
            solution, info = spla.gmres(A, b, tol=tolerance, maxiter=max_iter, callback=callback)
        elif solver_type == "bicgstab":
            solution, info = spla.bicgstab(A, b, tol=tolerance, maxiter=max_iter, callback=callback)
        else:
            raise ValueError(f"Unknown iterative solver: {solver_type}")
        
        self.solve_time = time.time() - start_time
        
        if info != 0:
            print(f"Warning: Iterative solver did not converge (info={info})")
        
        return solution
    
    def solve(self) -> np.ndarray:
        """Main solve method"""
        
        print("Building system matrix...")
        A, b = self.build_system_matrix()
        
        print(f"System size: {A.shape[0]} x {A.shape[1]}")
        print(f"Matrix sparsity: {A.nnz / (A.shape[0] * A.shape[1]) * 100:.2f}%")
        
        # Choose solver
        if self.solver_config.solver_type.lower() == "direct":
            print("Solving with direct method...")
            solution_vector = self.solve_direct(A, b)
        else:
            print(f"Solving with {self.solver_config.solver_type} iterative method...")
            solution_vector = self.solve_iterative(A, b)
        
        print(f"Solve time: {self.solve_time:.4f} seconds")
        
        # Reshape solution to 2D grid
        solution_2d = solution_vector.reshape((self.mesh.nx, self.mesh.ny))
        self.solution = solution_2d
        
        return solution_2d
    
    def get_solution_at_points(self, x_points: np.ndarray, y_points: np.ndarray) -> np.ndarray:
        """Interpolate solution at arbitrary points"""
        if self.solution is None:
            raise ValueError("No solution available. Call solve() first.")
        
        # Simple bilinear interpolation
        from scipy.interpolate import RegularGridInterpolator
        
        # Create interpolator
        x_coords = self.mesh.x_centers[:, 0]
        y_coords = self.mesh.y_centers[0, :]
        
        interpolator = RegularGridInterpolator(
            (x_coords, y_coords), 
            self.solution, 
            method='linear',
            bounds_error=False,
            fill_value=0.0
        )
        
        # Evaluate at points
        points = np.column_stack([x_points.ravel(), y_points.ravel()])
        values = interpolator(points)
        
        return values.reshape(x_points.shape)
    
    def compute_residual(self, solution: Optional[np.ndarray] = None) -> float:
        """Compute L2 norm of residual"""
        if solution is None:
            if self.solution is None:
                raise ValueError("No solution available")
            solution = self.solution.ravel()
        
        A, b = self.build_system_matrix()
        residual = A @ solution - b
        return np.linalg.norm(residual)
    
    def get_solver_info(self) -> dict:
        """Get information about the solve"""
        return {
            'solve_time': self.solve_time,
            'mesh_size': (self.mesh.nx, self.mesh.ny),
            'total_cells': self.mesh.total_cells,
            'solver_type': self.solver_config.solver_type,
            'residual_history': self.residual_history,
            'final_residual': self.residual_history[-1] if self.residual_history else None
        }

class FiniteVolumeSolverFactory:
    """Factory for creating finite volume solvers"""
    
    @staticmethod
    def create_solver(mesh, source_function, boundary_condition, solver_config):
        """Create appropriate solver based on configuration"""
        return FiniteVolumeSolver(mesh, source_function, boundary_condition, solver_config)