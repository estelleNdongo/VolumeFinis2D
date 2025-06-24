"""
Boundary conditions module for finite volume solver
"""
import numpy as np
from typing import Tuple, Dict, Any
from abc import ABC, abstractmethod

class BoundaryCondition(ABC):
    """Abstract base class for boundary conditions"""
    
    @abstractmethod
    def apply_to_matrix(self, A: np.ndarray, b: np.ndarray, mesh, boundary_info: Dict):
        """Apply boundary condition to the system matrix A and right-hand side b"""
        pass

class DirichletBC(BoundaryCondition):
    """Dirichlet boundary conditions (fixed values) - FIXED VERSION"""
    
    def __init__(self, values: Dict[str, np.ndarray]):
        self.values = values
    
    def apply_to_matrix(self, A: np.ndarray, b: np.ndarray, mesh, boundary_info: Dict):
        """Apply Dirichlet BC to system matrix - CORRECTED"""
        nx, ny = mesh.nx, mesh.ny
        
        def get_global_index(i: int, j: int) -> int:
            """Convert (i,j) cell indices to global matrix index"""
            return i * ny + j
        
        # Store boundary indices to apply Dirichlet conditions
        boundary_indices = set()
        
        # Apply boundary conditions
        for boundary, values in self.values.items():
            if boundary == 'left':  # i = 0
                for j in range(ny):
                    idx = get_global_index(0, j)
                    boundary_indices.add(idx)
                    self._apply_dirichlet_at_index(A, b, idx, values[j])
                    
            elif boundary == 'right':  # i = nx-1
                for j in range(ny):
                    idx = get_global_index(nx-1, j)
                    boundary_indices.add(idx)
                    self._apply_dirichlet_at_index(A, b, idx, values[j])
                    
            elif boundary == 'bottom':  # j = 0
                for i in range(nx):
                    idx = get_global_index(i, 0)
                    boundary_indices.add(idx)
                    self._apply_dirichlet_at_index(A, b, idx, values[i])
                    
            elif boundary == 'top':  # j = ny-1
                for i in range(nx):
                    idx = get_global_index(i, ny-1)
                    boundary_indices.add(idx)
                    self._apply_dirichlet_at_index(A, b, idx, values[i])
    
    def _apply_dirichlet_at_index(self, A: np.ndarray, b: np.ndarray, idx: int, value: float):
        """Apply Dirichlet BC at specific matrix index - CORRECTED"""
        # Zero out the entire row
        A[idx, :] = 0.0
        # Set diagonal to 1
        A[idx, idx] = 1.0
        # Set right-hand side to the boundary value
        b[idx] = value
        
        # CRITICAL FIX: Also zero out the column for this boundary node
        # and adjust the RHS for other equations
        for i in range(A.shape[0]):
            if i != idx and A[i, idx] != 0.0:
                # Subtract the contribution of the boundary node
                b[i] -= A[i, idx] * value
                # Zero out the matrix entry
                A[i, idx] = 0.0
    """Dirichlet boundary conditions (fixed values)"""
    
    def __init__(self, values: Dict[str, np.ndarray]):
        """
        Args:
            values: Dictionary with keys 'left', 'right', 'bottom', 'top'
                   and corresponding numpy arrays of boundary values
        """
        self.values = values
    
    def apply_to_matrix(self, A: np.ndarray, b: np.ndarray, mesh, boundary_info: Dict):
        """Apply Dirichlet BC to system matrix"""
        nx, ny = mesh.nx, mesh.ny
        
        def get_global_index(i: int, j: int) -> int:
            """Convert (i,j) cell indices to global matrix index"""
            return i * ny + j
        
        # Apply boundary conditions
        for boundary, values in self.values.items():
            if boundary == 'left':  # i = 0
                for j in range(ny):
                    idx = get_global_index(0, j)
                    self._apply_dirichlet_at_index(A, b, idx, values[j])
                    
            elif boundary == 'right':  # i = nx-1
                for j in range(ny):
                    idx = get_global_index(nx-1, j)
                    self._apply_dirichlet_at_index(A, b, idx, values[j])
                    
            elif boundary == 'bottom':  # j = 0
                for i in range(nx):
                    idx = get_global_index(i, 0)
                    self._apply_dirichlet_at_index(A, b, idx, values[i])
                    
            elif boundary == 'top':  # j = ny-1
                for i in range(nx):
                    idx = get_global_index(i, ny-1)
                    self._apply_dirichlet_at_index(A, b, idx, values[i])
    
    def _apply_dirichlet_at_index(self, A: np.ndarray, b: np.ndarray, idx: int, value: float):
        """Apply Dirichlet BC at specific matrix index"""
        # Set row to identity
        A[idx, :] = 0.0
        A[idx, idx] = 1.0
        b[idx] = value

class NeumannBC(BoundaryCondition):
    """Neumann boundary conditions (fixed gradient)"""
    
    def __init__(self, gradients: Dict[str, np.ndarray]):
        """
        Args:
            gradients: Dictionary with keys 'left', 'right', 'bottom', 'top'
                      and corresponding numpy arrays of normal gradient values
        """
        self.gradients = gradients
    
    def apply_to_matrix(self, A: np.ndarray, b: np.ndarray, mesh, boundary_info: Dict):
        """Apply Neumann BC to system matrix"""
        nx, ny = mesh.nx, mesh.ny
        
        def get_global_index(i: int, j: int) -> int:
            return i * ny + j
        
        for boundary, grad_values in self.gradients.items():
            if boundary == 'left':  # i = 0, normal points in -x direction
                for j in range(ny):
                    idx = get_global_index(0, j)
                    # Modify equation for ghost cell approach
                    # -∂u/∂x = grad_value => u_ghost = u_0 - dx * grad_value
                    dx = mesh.get_face_distance('west', 0, j)
                    face_area = mesh.get_face_area('west', 0, j)
                    
                    # Flux contribution: -grad_value * face_area
                    b[idx] += grad_values[j] * face_area
                    
            elif boundary == 'right':  # i = nx-1, normal points in +x direction
                for j in range(ny):
                    idx = get_global_index(nx-1, j)
                    dx = mesh.get_face_distance('east', nx-1, j)
                    face_area = mesh.get_face_area('east', nx-1, j)
                    
                    # Flux contribution: grad_value * face_area
                    b[idx] += grad_values[j] * face_area
                    
            elif boundary == 'bottom':  # j = 0, normal points in -y direction
                for i in range(nx):
                    idx = get_global_index(i, 0)
                    dy = mesh.get_face_distance('south', i, 0)
                    face_area = mesh.get_face_area('south', i, 0)
                    
                    # Flux contribution: -grad_value * face_area
                    b[idx] += grad_values[i] * face_area
                    
            elif boundary == 'top':  # j = ny-1, normal points in +y direction
                for i in range(nx):
                    idx = get_global_index(i, ny-1)
                    dy = mesh.get_face_distance('north', i, ny-1)
                    face_area = mesh.get_face_area('north', i, ny-1)
                    
                    # Flux contribution: grad_value * face_area
                    b[idx] += grad_values[i] * face_area

class RobinBC(BoundaryCondition):
    """Robin boundary conditions (mixed): α*u + β*∂u/∂n = γ"""
    
    def __init__(self, alpha: Dict[str, np.ndarray], 
                 beta: Dict[str, np.ndarray], 
                 gamma: Dict[str, np.ndarray]):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
    
    def apply_to_matrix(self, A: np.ndarray, b: np.ndarray, mesh, boundary_info: Dict):
        """Apply Robin BC to system matrix"""
        nx, ny = mesh.nx, mesh.ny
        
        def get_global_index(i: int, j: int) -> int:
            return i * ny + j
        
        for boundary in self.alpha.keys():
            alpha_vals = self.alpha[boundary]
            beta_vals = self.beta[boundary]
            gamma_vals = self.gamma[boundary]
            
            if boundary == 'left':
                for j in range(ny):
                    idx = get_global_index(0, j)
                    dx = mesh.get_face_distance('west', 0, j)
                    face_area = mesh.get_face_area('west', 0, j)
                    
                    # Robin BC: α*u + β*(-∂u/∂x) = γ
                    # Discretization: α*u_0 + β*(u_ghost - u_0)/dx = γ
                    # Eliminate ghost: α*u_0 - β*u_0/dx + β*γ*dx/β = γ
                    coeff = alpha_vals[j] - beta_vals[j] / dx
                    A[idx, idx] += coeff * face_area
                    b[idx] += gamma_vals[j] * face_area
                    
            # Similar for other boundaries...

class MixedBC(BoundaryCondition):
    """Mixed boundary conditions (different types on different boundaries)"""
    
    def __init__(self, bc_dict: Dict[str, BoundaryCondition]):
        """
        Args:
            bc_dict: Dictionary mapping boundary names to boundary condition objects
        """
        self.bc_dict = bc_dict
    
    def apply_to_matrix(self, A: np.ndarray, b: np.ndarray, mesh, boundary_info: Dict):
        """Apply mixed boundary conditions"""
        for boundary_name, bc in self.bc_dict.items():
            # Create temporary boundary info for single boundary
            temp_info = {boundary_name: boundary_info.get(boundary_name, {})}
            bc.apply_to_matrix(A, b, mesh, temp_info)

class BoundaryConditionFactory:
    """Factory class for creating boundary conditions"""
    
    @staticmethod
    def create_from_config(bc_config, boundary_values: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]):
        """Create boundary condition from configuration"""
        left_vals, right_vals, bottom_vals, top_vals = boundary_values
        
        bc_type = bc_config.bc_type.lower()
        
        if bc_type == "dirichlet":
            values = {
                'left': left_vals,
                'right': right_vals,
                'bottom': bottom_vals,
                'top': top_vals
            }
            return DirichletBC(values)
            
        elif bc_type == "neumann":
            gradients = {
                'left': left_vals,   # These would be gradient values in this case
                'right': right_vals,
                'bottom': bottom_vals,
                'top': top_vals
            }
            return NeumannBC(gradients)
            
        elif bc_type == "mixed":
            # Example: Dirichlet on left/right, Neumann on top/bottom
            bc_dict = {
                'left': DirichletBC({'left': left_vals}),
                'right': DirichletBC({'right': right_vals}),
                'bottom': NeumannBC({'bottom': bottom_vals}),
                'top': NeumannBC({'top': top_vals})
            }
            return MixedBC(bc_dict)
            
        elif bc_type == "robin":
            # Example Robin BC with α=1, β=1
            alpha = {'left': np.ones_like(left_vals), 'right': np.ones_like(right_vals),
                    'bottom': np.ones_like(bottom_vals), 'top': np.ones_like(top_vals)}
            beta = {'left': np.ones_like(left_vals), 'right': np.ones_like(right_vals),
                   'bottom': np.ones_like(bottom_vals), 'top': np.ones_like(top_vals)}
            gamma = {
                'left': left_vals,
                'right': right_vals,
                'bottom': bottom_vals,
                'top': top_vals
            }
            return RobinBC(alpha, beta, gamma)
            
        else:
            raise ValueError(f"Unsupported boundary condition type: {bc_type}")

def create_boundary_condition(bc_config, boundary_values):
    """Factory function for creating boundary conditions"""
    return BoundaryConditionFactory.create_from_config(bc_config, boundary_values)