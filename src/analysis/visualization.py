"""
Visualization module for finite volume solver results
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from mpl_toolkits.mplot3d import Axes3D
from typing import Dict, List, Optional, Tuple
import os

class ResultVisualizer:
    """Class for visualizing finite volume solver results"""
    
    def __init__(self, save_plots: bool = True, output_dir: str = "plots", dpi: int = 300):
        self.save_plots = save_plots
        self.output_dir = output_dir
        self.dpi = dpi
        
        # Create output directory if it doesn't exist
        if self.save_plots:
            os.makedirs(self.output_dir, exist_ok=True)
        
        # Set up matplotlib style
        plt.style.use('default')
        plt.rcParams.update({
            'font.size': 12,
            'axes.titlesize': 14,
            'axes.labelsize': 12,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'legend.fontsize': 11,
            'figure.titlesize': 16
        })
    
    def plot_solution_2d(self, solution: np.ndarray, mesh, 
                        title: str = "Numerical Solution",
                        colorbar_label: str = "u", 
                        filename: Optional[str] = None) -> plt.Figure:
        """Plot 2D solution as contour/heatmap"""
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Create contour plot
        X, Y = mesh.x_centers, mesh.y_centers
        levels = 20
        
        # Filled contours
        contourf = ax.contourf(X, Y, solution, levels=levels, cmap='viridis')
        
        # Contour lines
        contour = ax.contour(X, Y, solution, levels=levels, colors='black', alpha=0.3, linewidths=0.5)
        
        # Colorbar
        cbar = plt.colorbar(contourf, ax=ax, label=colorbar_label)
        
        # Labels and title
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title(title)
        ax.set_aspect('equal')
        
        # Grid
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if self.save_plots and filename:
            filepath = os.path.join(self.output_dir, f"{filename}.png")
            plt.savefig(filepath, dpi=self.dpi, bbox_inches='tight')
            print(f"Saved plot: {filepath}")
        
        return fig
    
    def plot_solution_3d(self, solution: np.ndarray, mesh,
                        title: str = "3D Solution",
                        filename: Optional[str] = None) -> plt.Figure:
        """Plot 3D surface of solution"""
        
        fig = plt.figure(figsize=(12, 9))
        ax = fig.add_subplot(111, projection='3d')
        
        X, Y = mesh.x_centers, mesh.y_centers
        
        # Surface plot
        surf = ax.plot_surface(X, Y, solution, cmap='viridis', 
                              alpha=0.9, edgecolor='none')
        
        # Colorbar
        cbar = plt.colorbar(surf, ax=ax, label='u', shrink=0.5)
        
        # Labels and title
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('u')
        ax.set_title(title)
        
        plt.tight_layout()
        
        if self.save_plots and filename:
            filepath = os.path.join(self.output_dir, f"{filename}.png")
            plt.savefig(filepath, dpi=self.dpi, bbox_inches='tight')
            print(f"Saved plot: {filepath}")
        
        return fig
    
    def plot_error_distribution(self, numerical_solution: np.ndarray,
                               exact_solution: np.ndarray, mesh,
                               title: str = "Error Distribution",
                               filename: Optional[str] = None) -> plt.Figure:
        """Plot error distribution between numerical and exact solutions"""
        
        error = np.abs(numerical_solution - exact_solution)
        max_error = np.max(error)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        X, Y = mesh.x_centers, mesh.y_centers
        
        # Error contour plot
        levels = 20
        contourf1 = ax1.contourf(X, Y, error, levels=levels, cmap='Reds')
        contour1 = ax1.contour(X, Y, error, levels=levels, colors='black', alpha=0.3, linewidths=0.5)
        cbar1 = plt.colorbar(contourf1, ax=ax1, label='|Error|')
        
        ax1.set_xlabel('x')
        ax1.set_ylabel('y')
        ax1.set_title(f'Absolute Error\nMax Error = {max_error:.2e}')
        ax1.set_aspect('equal')
        ax1.grid(True, alpha=0.3)
        
        # Error histogram
        error_flat = error.ravel()
        ax2.hist(error_flat, bins=50, density=True, alpha=0.7, color='red', edgecolor='black')
        ax2.set_xlabel('|Error|')
        ax2.set_ylabel('Density')
        ax2.set_title('Error Distribution')
        ax2.grid(True, alpha=0.3)
        
        # Add statistics
        mean_error = np.mean(error_flat)
        std_error = np.std(error_flat)
        ax2.axvline(mean_error, color='blue', linestyle='--', label=f'Mean: {mean_error:.2e}')
        ax2.axvline(mean_error + std_error, color='green', linestyle='--', label=f'Mean + Std: {mean_error + std_error:.2e}')
        ax2.legend()
        
        plt.tight_layout()
        
        if self.save_plots and filename:
            filepath = os.path.join(self.output_dir, f"{filename}.png")
            plt.savefig(filepath, dpi=self.dpi, bbox_inches='tight')
            print(f"Saved plot: {filepath}")
        
        return fig
    
    def plot_convergence_study(self, convergence_analysis: Dict[str, Dict],
                              title: str = "Convergence Study",
                              filename: Optional[str] = None) -> plt.Figure:
        """Plot convergence rates for different error norms"""
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Plot 1: Error vs mesh size
        for norm_name, analysis in convergence_analysis.items():
            if norm_name in ['exact_L2_norm']:
                continue
                
            mesh_sizes = analysis['mesh_sizes']
            errors = analysis['errors']
            rate = analysis['convergence_rate']
            
            # Plot actual errors
            ax1.loglog(mesh_sizes, errors, 'o-', label=f'{norm_name} (rate={rate:.2f})', linewidth=2, markersize=6)
            
            # Plot theoretical convergence line
            if len(mesh_sizes) >= 2:
                h_theory = np.array([min(mesh_sizes), max(mesh_sizes)])
                C = errors[0] / (mesh_sizes[0] ** analysis['theoretical_rate'])
                theory_errors = C * h_theory ** analysis['theoretical_rate']
                ax1.loglog(h_theory, theory_errors, '--', alpha=0.7, 
                          label=f'{norm_name} theory (rate=2)')
        
        ax1.set_xlabel('Mesh size h')
        ax1.set_ylabel('Error')
        ax1.set_title('Convergence Study')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Solve time vs mesh size
        if convergence_analysis:
            first_analysis = next(iter(convergence_analysis.values()))
            mesh_sizes = first_analysis['mesh_sizes']
            solve_times = first_analysis['solve_times']
            
            # Number of cells
            n_cells = [1.0 / h**2 for h in mesh_sizes]
            
            ax2.loglog(n_cells, solve_times, 'ro-', linewidth=2, markersize=6, label='Actual')
            
            # Theoretical scaling (should be roughly O(N) for direct solver)
            if len(n_cells) >= 2:
                C_time = solve_times[0] / n_cells[0]
                theory_times = [C_time * n for n in n_cells]
                ax2.loglog(n_cells, theory_times, 'r--', alpha=0.7, label='O(N)')
                
                # O(N^1.5) scaling for sparse direct solvers
                C_time_15 = solve_times[0] / (n_cells[0] ** 1.5)
                theory_times_15 = [C_time_15 * (n ** 1.5) for n in n_cells]
                ax2.loglog(n_cells, theory_times_15, 'g--', alpha=0.7, label='O(N^1.5)')
        
        ax2.set_xlabel('Number of cells')
        ax2.set_ylabel('Solve time (s)')
        ax2.set_title('Computational Efficiency')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if self.save_plots and filename:
            filepath = os.path.join(self.output_dir, f"{filename}.png")
            plt.savefig(filepath, dpi=self.dpi, bbox_inches='tight')
            print(f"Saved plot: {filepath}")
        
        return fig
    
    def plot_comparison(self, numerical_solution: np.ndarray,
                       exact_solution: np.ndarray, mesh,
                       title: str = "Solution Comparison",
                       filename: Optional[str] = None) -> plt.Figure:
        """Plot side-by-side comparison of numerical and exact solutions"""
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        X, Y = mesh.x_centers, mesh.y_centers
        
        # Numerical solution
        levels = 20
        im1 = axes[0, 0].contourf(X, Y, numerical_solution, levels=levels, cmap='viridis')
        axes[0, 0].set_title('Numerical Solution')
        axes[0, 0].set_xlabel('x')
        axes[0, 0].set_ylabel('y')
        axes[0, 0].set_aspect('equal')
        plt.colorbar(im1, ax=axes[0, 0], label='u')
        
        # Exact solution
        im2 = axes[0, 1].contourf(X, Y, exact_solution, levels=levels, cmap='viridis')
        axes[0, 1].set_title('Exact Solution')
        axes[0, 1].set_xlabel('x')
        axes[0, 1].set_ylabel('y')
        axes[0, 1].set_aspect('equal')
        plt.colorbar(im2, ax=axes[0, 1], label='u')
        
        # Error
        error = np.abs(numerical_solution - exact_solution)
        im3 = axes[1, 0].contourf(X, Y, error, levels=levels, cmap='Reds')
        axes[1, 0].set_title(f'Absolute Error\nMax Error = {np.max(error):.2e}')
        axes[1, 0].set_xlabel('x')
        axes[1, 0].set_ylabel('y')
        axes[1, 0].set_aspect('equal')
        plt.colorbar(im3, ax=axes[1, 0], label='|Error|')
        
        # Line plot comparison
        mid_j = mesh.ny // 2
        x_line = X[:, mid_j]
        num_line = numerical_solution[:, mid_j]
        exact_line = exact_solution[:, mid_j]
        
        axes[1, 1].plot(x_line, num_line, 'b-', linewidth=2, label='Numerical')
        axes[1, 1].plot(x_line, exact_line, 'r--', linewidth=2, label='Exact')
        axes[1, 1].set_title(f'Line Plot (y = {Y[0, mid_j]:.2f})')
        axes[1, 1].set_xlabel('x')
        axes[1, 1].set_ylabel('u')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if self.save_plots and filename:
            filepath = os.path.join(self.output_dir, f"{filename}.png")
            plt.savefig(filepath, dpi=self.dpi, bbox_inches='tight')
            print(f"Saved plot: {filepath}")
        
        return fig
    
    def plot_mesh(self, mesh, title: str = "Computational Mesh",
                  filename: Optional[str] = None) -> plt.Figure:
        """Plot the computational mesh"""
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Plot cell boundaries
        X_faces, Y_faces = mesh.x_faces, mesh.y_faces
        
        # Vertical lines
        for i in range(mesh.nx + 1):
            ax.plot(X_faces[i, :], Y_faces[i, :], 'k-', alpha=0.5, linewidth=0.5)
        
        # Horizontal lines
        for j in range(mesh.ny + 1):
            ax.plot(X_faces[:, j], Y_faces[:, j], 'k-', alpha=0.5, linewidth=0.5)
        
        # Plot cell centers
        ax.plot(mesh.x_centers.ravel(), mesh.y_centers.ravel(), 'ro', markersize=1, alpha=0.7)
        
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title(f'{title}\n{mesh.nx} × {mesh.ny} cells')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if self.save_plots and filename:
            filepath = os.path.join(self.output_dir, f"{filename}.png")
            plt.savefig(filepath, dpi=self.dpi, bbox_inches='tight')
            print(f"Saved plot: {filepath}")
        
        return fig

def create_visualizer(config) -> ResultVisualizer:
    """Factory function to create visualizer from configuration"""
    return ResultVisualizer(
        save_plots=config.visualization.save_plots,
        output_dir="plots",
        dpi=config.visualization.dpi
    )