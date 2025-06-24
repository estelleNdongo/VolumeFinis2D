"""
Configuration module for finite volume solver
"""
import numpy as np
from dataclasses import dataclass
from typing import Tuple, Callable, Dict, Any

@dataclass
class DomainConfig:
    """Configuration for computational domain"""
    x_min: float = 0.0
    x_max: float = 1.0
    y_min: float = 0.0
    y_max: float = 1.0
    
    @property
    def size(self) -> Tuple[float, float]:
        return (self.x_max - self.x_min, self.y_max - self.y_min)

@dataclass
class MeshConfig:
    """Configuration for mesh generation"""
    nx: int = 20  # Number of cells in x-direction
    ny: int = 20  # Number of cells in y-direction
    mesh_type: str = "uniform"  # "uniform", "refined"
    
    @property
    def total_cells(self) -> int:
        return self.nx * self.ny

@dataclass
class BoundaryConfig:
    """Configuration for boundary conditions"""
    bc_type: str = "dirichlet"  # "dirichlet", "neumann", "mixed"
    left_value: float = 0.0
    right_value: float = 0.0
    bottom_value: float = 0.0
    top_value: float = 0.0
    
    # For Neumann conditions (gradient values)
    left_grad: float = 0.0
    right_grad: float = 0.0
    bottom_grad: float = 0.0
    top_grad: float = 0.0

@dataclass
class SolverConfig:
    """Configuration for numerical solver"""
    solver_type: str = "direct"  # "direct", "iterative"
    tolerance: float = 1e-10
    max_iterations: int = 1000
    preconditioner: str = "none"  # "none", "jacobi", "ilu"

@dataclass
class AnalysisConfig:
    """Configuration for error analysis and convergence studies"""
    compute_error: bool = True
    convergence_study: bool = True
    mesh_refinement_levels: list = None
    error_norms: list = None
    
    def __post_init__(self):
        if self.mesh_refinement_levels is None:
            self.mesh_refinement_levels = [10, 20, 40, 80]
        if self.error_norms is None:
            self.error_norms = ["L2", "Linf"]

@dataclass
class VisualizationConfig:
    """Configuration for visualization"""
    plot_solution: bool = True
    plot_error: bool = True
    plot_convergence: bool = True
    save_plots: bool = True
    plot_format: str = "png"  # "png", "pdf", "svg"
    dpi: int = 300

class ProblemConfig:
    """Main configuration class that combines all configurations"""
    
    def __init__(self):
        self.domain = DomainConfig()
        self.mesh = MeshConfig()
        self.boundary = BoundaryConfig()
        self.solver = SolverConfig()
        self.analysis = AnalysisConfig()
        self.visualization = VisualizationConfig()
        
        # Function configurations
        self.function_name = "quadratic"
        self.custom_functions = {}
    
    def update_from_dict(self, config_dict: Dict[str, Any]):
        """Update configuration from dictionary"""
        for section, values in config_dict.items():
            if hasattr(self, section):
                section_obj = getattr(self, section)
                for key, value in values.items():
                    if hasattr(section_obj, key):
                        setattr(section_obj, key, value)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return {
            "domain": self.domain.__dict__,
            "mesh": self.mesh.__dict__,
            "boundary": self.boundary.__dict__,
            "solver": self.solver.__dict__,
            "analysis": self.analysis.__dict__,
            "visualization": self.visualization.__dict__,
            "function_name": self.function_name
        }
    
    @classmethod
    def from_args(cls, args):
        """Create configuration from command line arguments"""
        config = cls()
        
        # Update mesh configuration
        if hasattr(args, 'nx'):
            config.mesh.nx = args.nx
        if hasattr(args, 'ny'):
            config.mesh.ny = args.ny
            
        # Update function configuration
        if hasattr(args, 'function'):
            config.function_name = args.function
            
        # Update boundary configuration
        if hasattr(args, 'bc_type'):
            config.boundary.bc_type = args.bc_type
            
        return config

# Predefined configurations for common test cases
PREDEFINED_CONFIGS = {
    "simple_quadratic": {
        "domain": {"x_min": 0.0, "x_max": 1.0, "y_min": 0.0, "y_max": 1.0},
        "mesh": {"nx": 20, "ny": 20},
        "boundary": {"bc_type": "dirichlet"},
        "function_name": "quadratic"
    },
    
    "convergence_study": {
        "domain": {"x_min": 0.0, "x_max": 1.0, "y_min": 0.0, "y_max": 1.0},
        "analysis": {
            "convergence_study": True,
            "mesh_refinement_levels": [10, 20, 40, 80, 160]
        },
        "function_name": "manufactured"
    },
    
    "mixed_boundary": {
        "domain": {"x_min": 0.0, "x_max": 1.0, "y_min": 0.0, "y_max": 1.0},
        "mesh": {"nx": 40, "ny": 40},
        "boundary": {"bc_type": "mixed"},
        "function_name": "trigonometric"
    }
}