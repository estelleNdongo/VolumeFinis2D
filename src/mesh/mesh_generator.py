"""
Mesh generation module for finite volume solver
"""
import numpy as np
from typing import Tuple, List
from dataclasses import dataclass

@dataclass
class Mesh2D:
    """2D mesh data structure for finite volumes"""
    
    # Cell centers
    x_centers: np.ndarray  # (nx, ny)
    y_centers: np.ndarray  # (nx, ny)
    
    # Cell faces coordinates
    x_faces: np.ndarray    # (nx+1, ny+1)
    y_faces: np.ndarray    # (nx+1, ny+1)
    
    # Mesh spacing
    dx: np.ndarray         # (nx,) or scalar
    dy: np.ndarray         # (ny,) or scalar
    
    # Mesh dimensions
    nx: int
    ny: int
    
    # Domain bounds
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    
    @property
    def total_cells(self) -> int:
        return self.nx * self.ny
    
    @property
    def is_uniform(self) -> bool:
        """Check if mesh is uniform"""
        dx_uniform = np.allclose(self.dx, self.dx[0]) if hasattr(self.dx, '__len__') else True
        dy_uniform = np.allclose(self.dy, self.dy[0]) if hasattr(self.dy, '__len__') else True
        return dx_uniform and dy_uniform
    
    def get_cell_volume(self, i: int, j: int) -> float:
        """Get volume (area) of cell (i,j)"""
        dx_i = self.dx[i] if hasattr(self.dx, '__len__') else self.dx
        dy_j = self.dy[j] if hasattr(self.dy, '__len__') else self.dy
        return dx_i * dy_j
    
    def get_face_area(self, face_type: str, i: int, j: int) -> float:
        """Get area of face (length in 2D)"""
        if face_type in ['east', 'west']:
            return self.dy[j] if hasattr(self.dy, '__len__') else self.dy
        elif face_type in ['north', 'south']:
            return self.dx[i] if hasattr(self.dx, '__len__') else self.dx
        else:
            raise ValueError(f"Unknown face type: {face_type}")
    
    def get_face_distance(self, face_type: str, i: int, j: int) -> float:
        """Get distance between cell centers across a face"""
        if face_type == 'east':
            if i < self.nx - 1:
                return 0.5 * (self.dx[i] + self.dx[i+1]) if hasattr(self.dx, '__len__') else self.dx
            else:
                return 0.5 * self.dx[i] if hasattr(self.dx, '__len__') else 0.5 * self.dx
        elif face_type == 'west':
            if i > 0:
                return 0.5 * (self.dx[i-1] + self.dx[i]) if hasattr(self.dx, '__len__') else self.dx
            else:
                return 0.5 * self.dx[i] if hasattr(self.dx, '__len__') else 0.5 * self.dx
        elif face_type == 'north':
            if j < self.ny - 1:
                return 0.5 * (self.dy[j] + self.dy[j+1]) if hasattr(self.dy, '__len__') else self.dy
            else:
                return 0.5 * self.dy[j] if hasattr(self.dy, '__len__') else 0.5 * self.dy
        elif face_type == 'south':
            if j > 0:
                return 0.5 * (self.dy[j-1] + self.dy[j]) if hasattr(self.dy, '__len__') else self.dy
            else:
                return 0.5 * self.dy[j] if hasattr(self.dy, '__len__') else 0.5 * self.dy
        else:
            raise ValueError(f"Unknown face type: {face_type}")

class MeshGenerator:
    """Mesh generator for various mesh types"""
    
    @staticmethod
    def create_uniform_mesh(nx: int, ny: int, x_min: float, x_max: float, 
                           y_min: float, y_max: float) -> Mesh2D:
        """Create uniform rectangular mesh"""
        
        # Calculate uniform spacing
        dx = (x_max - x_min) / nx
        dy = (y_max - y_min) / ny
        
        # Create face coordinates
        x_faces = np.linspace(x_min, x_max, nx + 1)
        y_faces = np.linspace(y_min, y_max, ny + 1)
        X_faces, Y_faces = np.meshgrid(x_faces, y_faces, indexing='ij')
        
        # Create cell center coordinates
        x_centers = x_faces[:-1] + dx / 2
        y_centers = y_faces[:-1] + dy / 2
        X_centers, Y_centers = np.meshgrid(x_centers, y_centers, indexing='ij')
        
        return Mesh2D(
            x_centers=X_centers,
            y_centers=Y_centers,
            x_faces=X_faces,
            y_faces=Y_faces,
            dx=dx,
            dy=dy,
            nx=nx,
            ny=ny,
            x_min=x_min,
            x_max=x_max,
            y_min=y_min,
            y_max=y_max
        )
    
    @staticmethod
    def create_refined_mesh(nx: int, ny: int, x_min: float, x_max: float,
                           y_min: float, y_max: float, refinement_zones: List[dict] = None) -> Mesh2D:
        """Create mesh with local refinement"""
        
        if refinement_zones is None:
            # Default: refine center region
            refinement_zones = [{
                'x_range': (0.25, 0.75),
                'y_range': (0.25, 0.75),
                'factor': 2.0
            }]
        
        # Start with base spacing
        base_dx = (x_max - x_min) / nx
        base_dy = (y_max - y_min) / ny
        
        # Create non-uniform spacing arrays
        dx_array = np.full(nx, base_dx)
        dy_array = np.full(ny, base_dy)
        
        # Apply refinements
        for zone in refinement_zones:
            x_range = zone['x_range']
            y_range = zone['y_range']
            factor = zone['factor']
            
            # Find cells in refinement zone
            x_centers_base = np.linspace(x_min + base_dx/2, x_max - base_dx/2, nx)
            y_centers_base = np.linspace(y_min + base_dy/2, y_max - base_dy/2, ny)
            
            x_mask = (x_centers_base >= x_range[0]) & (x_centers_base <= x_range[1])
            y_mask = (y_centers_base >= y_range[0]) & (y_centers_base <= y_range[1])
            
            dx_array[x_mask] /= factor
            dy_array[y_mask] /= factor
        
        # Create face coordinates from spacing
        x_faces = np.zeros(nx + 1)
        x_faces[0] = x_min
        for i in range(nx):
            x_faces[i + 1] = x_faces[i] + dx_array[i]
        
        y_faces = np.zeros(ny + 1)
        y_faces[0] = y_min
        for j in range(ny):
            y_faces[j + 1] = y_faces[j] + dy_array[j]
        
        X_faces, Y_faces = np.meshgrid(x_faces, y_faces, indexing='ij')
        
        # Create cell centers
        x_centers = x_faces[:-1] + dx_array / 2
        y_centers = y_faces[:-1] + dy_array / 2
        X_centers, Y_centers = np.meshgrid(x_centers, y_centers, indexing='ij')
        
        return Mesh2D(
            x_centers=X_centers,
            y_centers=Y_centers,
            x_faces=X_faces,
            y_faces=Y_faces,
            dx=dx_array,
            dy=dy_array,
            nx=nx,
            ny=ny,
            x_min=x_min,
            x_max=x_max,
            y_min=y_min,
            y_max=y_max
        )
    
    @staticmethod
    def create_stretched_mesh(nx: int, ny: int, x_min: float, x_max: float,
                             y_min: float, y_max: float, stretch_factor: float = 1.1) -> Mesh2D:
        """Create stretched mesh (geometric progression)"""
        
        def geometric_spacing(n: int, length: float, factor: float):
            """Create geometric spacing"""
            if abs(factor - 1.0) < 1e-10:
                return np.full(n, length / n)
            
            # Solve for first cell size
            r = factor
            s = (1 - r**n) / (1 - r)
            dx0 = length / s
            
            spacing = np.zeros(n)
            spacing[0] = dx0
            for i in range(1, n):
                spacing[i] = spacing[i-1] * r
            
            return spacing
        
        dx_array = geometric_spacing(nx, x_max - x_min, stretch_factor)
        dy_array = geometric_spacing(ny, y_max - y_min, stretch_factor)
        
        # Create faces from spacing
        x_faces = np.zeros(nx + 1)
        x_faces[0] = x_min
        for i in range(nx):
            x_faces[i + 1] = x_faces[i] + dx_array[i]
        
        y_faces = np.zeros(ny + 1)
        y_faces[0] = y_min
        for j in range(ny):
            y_faces[j + 1] = y_faces[j] + dy_array[j]
        
        X_faces, Y_faces = np.meshgrid(x_faces, y_faces, indexing='ij')
        
        # Create centers
        x_centers = x_faces[:-1] + dx_array / 2
        y_centers = y_faces[:-1] + dy_array / 2
        X_centers, Y_centers = np.meshgrid(x_centers, y_centers, indexing='ij')
        
        return Mesh2D(
            x_centers=X_centers,
            y_centers=Y_centers,
            x_faces=X_faces,
            y_faces=Y_faces,
            dx=dx_array,
            dy=dy_array,
            nx=nx,
            ny=ny,
            x_min=x_min,
            x_max=x_max,
            y_min=y_min,
            y_max=y_max
        )

def create_mesh(mesh_config, domain_config) -> Mesh2D:
    """Factory function to create mesh based on configuration"""
    
    mesh_type = mesh_config.mesh_type.lower()
    
    if mesh_type == "uniform":
        return MeshGenerator.create_uniform_mesh(
            mesh_config.nx, mesh_config.ny,
            domain_config.x_min, domain_config.x_max,
            domain_config.y_min, domain_config.y_max
        )
    elif mesh_type == "refined":
        return MeshGenerator.create_refined_mesh(
            mesh_config.nx, mesh_config.ny,
            domain_config.x_min, domain_config.x_max,
            domain_config.y_min, domain_config.y_max
        )
    elif mesh_type == "stretched":
        return MeshGenerator.create_stretched_mesh(
            mesh_config.nx, mesh_config.ny,
            domain_config.x_min, domain_config.x_max,
            domain_config.y_min, domain_config.y_max
        )
    else:
        raise ValueError(f"Unknown mesh type: {mesh_type}")