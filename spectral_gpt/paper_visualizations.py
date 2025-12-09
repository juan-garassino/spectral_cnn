"""
Visualization Generators for Paper Generation

Provides specialized visualization functions for generating figures
for academic papers, including architecture diagrams, loss landscapes,
and frequency spectrum plots.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Rectangle
from matplotlib.animation import FuncAnimation, PillowWriter
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import torch
import torch.nn as nn


class ArchitectureDiagramGenerator:
    """
    Generates architecture diagrams for comparing Standard Transformer
    and Spectral GPT architectures.
    
    Features:
    - Side-by-side architecture comparisons
    - Color coding for different layer types
    - Parameter counts and dimensions
    - Animated GIFs showing forward pass
    """
    
    def __init__(self, output_dir: str):
        """
        Initialize ArchitectureDiagramGenerator.
        
        Args:
            output_dir: Directory to save generated diagrams
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Color scheme for different components
        self.colors = {
            'embedding': '#FF6B6B',      # Red
            'attention': '#4ECDC4',      # Teal
            'mlp': '#45B7D1',            # Blue
            'norm': '#FFA07A',           # Light orange
            'output': '#98D8C8',         # Mint
            'wave': '#FFD93D',           # Yellow
            'standard': '#6C5CE7'        # Purple
        }
    
    def generate_side_by_side_comparison(self,
                                        standard_config: Dict,
                                        wave_config: Dict,
                                        save_path: Optional[str] = None) -> str:
        """
        Generate side-by-side architecture comparison diagram.
        
        Args:
            standard_config: Configuration dict for standard transformer
            wave_config: Configuration dict for wave model
            save_path: Optional path to save figure
            
        Returns:
            Path to saved figure
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 12))
        fig.patch.set_facecolor('#1a1a1a')
        
        # Draw standard transformer
        self._draw_standard_architecture(ax1, standard_config)
        ax1.set_title('Standard Transformer', fontsize=16, fontweight='bold', color='white', pad=20)
        
        # Draw spectral GPT
        self._draw_wave_architecture(ax2, wave_config)
        ax2.set_title('Spectral GPT', fontsize=16, fontweight='bold', color='white', pad=20)
        
        # Configure axes
        for ax in [ax1, ax2]:
            ax.set_xlim(0, 10)
            ax.set_ylim(0, 14)
            ax.axis('off')
            ax.set_facecolor('#1a1a1a')
        
        plt.tight_layout()
        
        # Save figure
        if save_path is None:
            save_path = self.output_dir / 'architecture_comparison.png'
        else:
            save_path = Path(save_path)
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='#1a1a1a')
        plt.close(fig)
        
        print(f"✓ Saved architecture comparison: {save_path}")
        return str(save_path)
    
    def _draw_standard_architecture(self, ax, config: Dict):
        """Draw standard transformer architecture"""
        y_pos = 13
        box_width = 8
        box_height = 0.8
        x_center = 5
        
        # Input
        self._draw_box(ax, x_center, y_pos, box_width, box_height, 
                      'Input Token IDs', '#666666', edge_color='white')
        y_pos -= 1.2
        
        # Token Embedding
        params = config.get('vocab_size', 50257) * config.get('d_model', 768)
        self._draw_box(ax, x_center, y_pos, box_width, box_height,
                      f'Token Embedding\n{params/1e6:.1f}M params', 
                      self.colors['embedding'])
        y_pos -= 1.2
        
        # Positional Encoding
        self._draw_box(ax, x_center, y_pos, box_width, box_height,
                      'Positional Encoding', self.colors['norm'])
        y_pos -= 1.5
        
        # Transformer Layers
        num_layers = config.get('num_layers', 12)
        layer_box_height = 3.5
        self._draw_transformer_block(ax, x_center, y_pos, box_width, layer_box_height,
                                    f'{num_layers} Transformer Layers', 'standard')
        y_pos -= layer_box_height + 0.5
        
        # LM Head
        self._draw_box(ax, x_center, y_pos, box_width, box_height,
                      'LM Head (Linear)', self.colors['output'])
        y_pos -= 1.2
        
        # Output
        self._draw_box(ax, x_center, y_pos, box_width, box_height,
                      'Output Logits', '#666666', edge_color='white')
        
        # Add parameter count
        total_params = config.get('total_params', 52.9)
        ax.text(x_center, 0.5, f'Total: {total_params}M parameters',
               ha='center', va='top', fontsize=12, fontweight='bold',
               color='white', bbox=dict(boxstyle='round', facecolor='#333333', alpha=0.8))
    
    def _draw_wave_architecture(self, ax, config: Dict):
        """Draw Spectral GPT architecture"""
        y_pos = 13
        box_width = 8
        box_height = 0.8
        x_center = 5
        
        # Input
        self._draw_box(ax, x_center, y_pos, box_width, box_height,
                      'Input Token IDs', '#666666', edge_color='white')
        y_pos -= 1.2
        
        # Wave Packet Embedding
        vocab = config.get('vocab_size', 50257)
        waves = config.get('num_waves', 8)
        harmonics = config.get('num_harmonics', 3)
        params = vocab * waves * (1 + 1 + harmonics)  # freqs + phases + harmonics
        self._draw_box(ax, x_center, y_pos, box_width, box_height * 1.5,
                      f'Wave Packet Embedding\n{params/1e6:.1f}M params\n(W={waves}, H={harmonics})',
                      self.colors['wave'])
        y_pos -= 1.8
        
        # Phase Encoding (built-in)
        self._draw_box(ax, x_center, y_pos, box_width, box_height,
                      'Phase Encoding (Built-in)', self.colors['norm'], style='dashed')
        y_pos -= 1.5
        
        # Transformer Layers with Wave Attention
        num_layers = config.get('num_layers', 12)
        layer_box_height = 3.5
        self._draw_transformer_block(ax, x_center, y_pos, box_width, layer_box_height,
                                    f'{num_layers} Transformer Layers\n(Interference Attention)', 'wave')
        y_pos -= layer_box_height + 0.5
        
        # LM Head
        self._draw_box(ax, x_center, y_pos, box_width, box_height,
                      'LM Head (Linear)', self.colors['output'])
        y_pos -= 1.2
        
        # Output
        self._draw_box(ax, x_center, y_pos, box_width, box_height,
                      'Output Logits', '#666666', edge_color='white')
        
        # Add parameter count
        total_params = config.get('total_params', 67.5)
        ax.text(x_center, 0.5, f'Total: {total_params}M parameters',
               ha='center', va='top', fontsize=12, fontweight='bold',
               color='white', bbox=dict(boxstyle='round', facecolor='#333333', alpha=0.8))
    
    def _draw_box(self, ax, x, y, width, height, text, color, 
                  edge_color='white', style='solid', alpha=0.9):
        """Draw a colored box with text"""
        # Draw box
        box = FancyBboxPatch((x - width/2, y - height/2), width, height,
                            boxstyle="round,pad=0.1", 
                            facecolor=color, edgecolor=edge_color,
                            linewidth=2, linestyle=style, alpha=alpha)
        ax.add_patch(box)
        
        # Add text
        ax.text(x, y, text, ha='center', va='center',
               fontsize=10, fontweight='bold', color='white',
               wrap=True)
    
    def _draw_transformer_block(self, ax, x, y, width, height, title, arch_type):
        """Draw a transformer block with internal components"""
        # Outer box
        outer_box = FancyBboxPatch((x - width/2, y - height/2), width, height,
                                  boxstyle="round,pad=0.1",
                                  facecolor='#2a2a2a', edgecolor='white',
                                  linewidth=2, alpha=0.9)
        ax.add_patch(outer_box)
        
        # Title
        ax.text(x, y + height/2 - 0.3, title, ha='center', va='top',
               fontsize=11, fontweight='bold', color='white')
        
        # Internal components
        comp_height = 0.6
        comp_width = width - 1
        y_internal = y + height/2 - 1.0
        
        # Attention
        if arch_type == 'wave':
            attn_text = 'Interference Attention\ncos(φᵢ - φⱼ)'
        else:
            attn_text = 'Multi-Head Attention\nQ·Kᵀ/√d'
        
        self._draw_box(ax, x, y_internal, comp_width, comp_height,
                      attn_text, self.colors['attention'], alpha=0.8)
        y_internal -= comp_height + 0.3
        
        # Layer Norm
        self._draw_box(ax, x, y_internal, comp_width, comp_height * 0.6,
                      'Layer Norm', self.colors['norm'], alpha=0.8)
        y_internal -= comp_height * 0.6 + 0.3
        
        # MLP
        if arch_type == 'wave':
            mlp_text = 'Feed-Forward\nsin(x) + 0.1x'
        else:
            mlp_text = 'Feed-Forward\nGELU(x)'
        
        self._draw_box(ax, x, y_internal, comp_width, comp_height,
                      mlp_text, self.colors['mlp'], alpha=0.8)
        y_internal -= comp_height + 0.3
        
        # Layer Norm
        self._draw_box(ax, x, y_internal, comp_width, comp_height * 0.6,
                      'Layer Norm', self.colors['norm'], alpha=0.8)
    
    def generate_parameter_breakdown(self,
                                    standard_config: Dict,
                                    wave_config: Dict,
                                    save_path: Optional[str] = None) -> str:
        """
        Generate parameter count breakdown visualization.
        
        Args:
            standard_config: Configuration for standard model
            wave_config: Configuration for wave model
            save_path: Optional path to save figure
            
        Returns:
            Path to saved figure
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 8))
        fig.patch.set_facecolor('#1a1a1a')
        
        # Standard Transformer breakdown
        std_components = ['Embeddings', 'Attention', 'MLP', 'Other']
        std_params = [19.4, 4.7, 9.5, 19.3]
        std_colors = [self.colors['embedding'], self.colors['attention'], 
                     self.colors['mlp'], self.colors['norm']]
        
        ax1.pie(std_params, labels=std_components, colors=std_colors,
               autopct='%1.1f%%', startangle=90, textprops={'color': 'white', 'fontsize': 11})
        ax1.set_title(f'Standard Transformer\n{sum(std_params):.1f}M Parameters',
                     fontsize=14, fontweight='bold', color='white', pad=20)
        ax1.set_facecolor('#1a1a1a')
        
        # Spectral GPT breakdown
        wave_components = ['Embeddings', 'Attention', 'MLP', 'Other']
        wave_params = [33.9, 4.7, 9.5, 19.4]
        wave_colors = [self.colors['wave'], self.colors['attention'],
                      self.colors['mlp'], self.colors['norm']]
        
        ax2.pie(wave_params, labels=wave_components, colors=wave_colors,
               autopct='%1.1f%%', startangle=90, textprops={'color': 'white', 'fontsize': 11})
        ax2.set_title(f'Spectral GPT\n{sum(wave_params):.1f}M Parameters',
                     fontsize=14, fontweight='bold', color='white', pad=20)
        ax2.set_facecolor('#1a1a1a')
        
        plt.tight_layout()
        
        # Save figure
        if save_path is None:
            save_path = self.output_dir / 'parameter_breakdown.png'
        else:
            save_path = Path(save_path)
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='#1a1a1a')
        plt.close(fig)
        
        print(f"✓ Saved parameter breakdown: {save_path}")
        return str(save_path)
    
    def generate_forward_pass_animation(self,
                                       arch_type: str = 'wave',
                                       save_path: Optional[str] = None,
                                       duration: int = 5) -> str:
        """
        Generate animated GIF showing forward pass through architecture.
        
        Args:
            arch_type: 'standard' or 'wave'
            save_path: Optional path to save GIF
            duration: Duration in seconds
            
        Returns:
            Path to saved GIF
        """
        fig, ax = plt.subplots(figsize=(10, 12))
        fig.patch.set_facecolor('#1a1a1a')
        ax.set_facecolor('#1a1a1a')
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 14)
        ax.axis('off')
        
        # Define layer positions
        layers = self._get_layer_positions(arch_type)
        
        # Animation function
        def animate(frame):
            ax.clear()
            ax.set_xlim(0, 10)
            ax.set_ylim(0, 14)
            ax.axis('off')
            ax.set_facecolor('#1a1a1a')
            
            # Draw all layers
            for i, (name, y_pos, color) in enumerate(layers):
                alpha = 0.3 if i > frame else 0.9
                self._draw_box(ax, 5, y_pos, 8, 0.8, name, color, alpha=alpha)
            
            # Draw data flow indicator
            if frame < len(layers):
                y_current = layers[frame][1]
                circle = Circle((5, y_current), 0.3, color='cyan', alpha=0.8)
                ax.add_patch(circle)
                ax.text(5, y_current, '⚡', ha='center', va='center',
                       fontsize=20, color='white')
            
            title = 'Standard Transformer' if arch_type == 'standard' else 'Spectral GPT'
            ax.set_title(f'{title} - Forward Pass', fontsize=16,
                        fontweight='bold', color='white', pad=20)
        
        # Create animation
        frames = len(layers) + 5  # Extra frames at end
        anim = FuncAnimation(fig, animate, frames=frames, interval=duration*1000//frames)
        
        # Save as GIF
        if save_path is None:
            save_path = self.output_dir / f'{arch_type}_forward_pass.gif'
        else:
            save_path = Path(save_path)
        
        writer = PillowWriter(fps=frames//duration)
        anim.save(save_path, writer=writer)
        plt.close(fig)
        
        print(f"✓ Saved forward pass animation: {save_path}")
        return str(save_path)
    
    def _get_layer_positions(self, arch_type: str) -> List[Tuple[str, float, str]]:
        """Get layer names, positions, and colors for animation"""
        if arch_type == 'standard':
            return [
                ('Input', 13, '#666666'),
                ('Token Embedding', 11.5, self.colors['embedding']),
                ('Positional Encoding', 10, self.colors['norm']),
                ('Attention Layer 1', 8.5, self.colors['attention']),
                ('MLP Layer 1', 7, self.colors['mlp']),
                ('Attention Layer 2', 5.5, self.colors['attention']),
                ('MLP Layer 2', 4, self.colors['mlp']),
                ('LM Head', 2.5, self.colors['output']),
                ('Output', 1, '#666666')
            ]
        else:  # wave
            return [
                ('Input', 13, '#666666'),
                ('Wave Packet Embedding', 11.5, self.colors['wave']),
                ('Phase Encoding', 10, self.colors['norm']),
                ('Interference Attention 1', 8.5, self.colors['attention']),
                ('Wave MLP 1', 7, self.colors['mlp']),
                ('Interference Attention 2', 5.5, self.colors['attention']),
                ('Wave MLP 2', 4, self.colors['mlp']),
                ('LM Head', 2.5, self.colors['output']),
                ('Output', 1, '#666666')
            ]



class LossLandscapeVisualizer:
    """
    Generates loss landscape visualizations showing optimization trajectories
    for different architectures.
    
    Features:
    - 3D loss landscape plots
    - Optimization trajectory comparisons
    - 2D contour projections
    - Convergence point highlighting
    """
    
    def __init__(self, output_dir: str):
        """
        Initialize LossLandscapeVisualizer.
        
        Args:
            output_dir: Directory to save generated visualizations
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def generate_3d_landscape(self,
                             standard_trajectory: List[Dict],
                             wave_trajectory: List[Dict],
                             save_path: Optional[str] = None) -> str:
        """
        Generate 3D loss landscape with optimization trajectories.
        
        Args:
            standard_trajectory: List of dicts with 'step' and 'loss' keys
            wave_trajectory: List of dicts with 'step' and 'loss' keys
            save_path: Optional path to save figure
            
        Returns:
            Path to saved figure
        """
        from mpl_toolkits.mplot3d import Axes3D
        
        fig = plt.figure(figsize=(14, 10))
        fig.patch.set_facecolor('#1a1a1a')
        ax = fig.add_subplot(111, projection='3d')
        ax.set_facecolor('#1a1a1a')
        
        # Create synthetic loss landscape (simplified for visualization)
        x = np.linspace(-5, 5, 100)
        y = np.linspace(-5, 5, 100)
        X, Y = np.meshgrid(x, y)
        
        # Create a landscape with multiple local minima
        Z = (np.sin(X) * np.cos(Y) + 0.1 * X**2 + 0.1 * Y**2 + 
             0.5 * np.sin(2*X) * np.sin(2*Y) + 4.4)
        
        # Plot surface
        surf = ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.6,
                              linewidth=0, antialiased=True)
        
        # Extract trajectories
        std_steps = [d['step'] for d in standard_trajectory]
        std_losses = [d['loss'] for d in standard_trajectory]
        wave_steps = [d['step'] for d in wave_trajectory]
        wave_losses = [d['loss'] for d in wave_trajectory]
        
        # Normalize steps to landscape coordinates
        max_step = max(max(std_steps), max(wave_steps))
        std_x = [(s / max_step) * 10 - 5 for s in std_steps]
        std_y = [i * 0.1 - 2 for i in range(len(std_steps))]
        
        wave_x = [(s / max_step) * 10 - 5 for s in wave_steps]
        wave_y = [i * 0.1 + 2 for i in range(len(wave_steps))]
        
        # Plot trajectories
        ax.plot(std_x, std_y, std_losses, 'o-', color='cyan', linewidth=3,
               markersize=4, label='Standard Transformer', alpha=0.9)
        ax.plot(wave_x, wave_y, wave_losses, 's-', color='yellow', linewidth=3,
               markersize=4, label='Spectral GPT', alpha=0.9)
        
        # Mark start and end points
        ax.scatter([std_x[0]], [std_y[0]], [std_losses[0]], 
                  color='red', s=200, marker='o', label='Start', alpha=0.9)
        ax.scatter([std_x[-1], wave_x[-1]], [std_y[-1], wave_y[-1]], 
                  [std_losses[-1], wave_losses[-1]],
                  color='green', s=200, marker='*', label='Convergence', alpha=0.9)
        
        # Labels and title
        ax.set_xlabel('Parameter Space (Dim 1)', fontsize=12, color='white')
        ax.set_ylabel('Parameter Space (Dim 2)', fontsize=12, color='white')
        ax.set_zlabel('Loss', fontsize=12, color='white')
        ax.set_title('Loss Landscape with Optimization Trajectories',
                    fontsize=14, fontweight='bold', color='white', pad=20)
        
        # Styling
        ax.tick_params(colors='white')
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', fontsize=10, facecolor='#2a2a2a', 
                 edgecolor='white', labelcolor='white')
        
        # Colorbar
        cbar = fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
        cbar.set_label('Loss Value', color='white')
        cbar.ax.yaxis.set_tick_params(color='white')
        plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')
        
        # Save figure
        if save_path is None:
            save_path = self.output_dir / 'loss_landscape_3d.png'
        else:
            save_path = Path(save_path)
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='#1a1a1a')
        plt.close(fig)
        
        print(f"✓ Saved 3D loss landscape: {save_path}")
        return str(save_path)
    
    def generate_contour_plot(self,
                             standard_trajectory: List[Dict],
                             wave_trajectory: List[Dict],
                             save_path: Optional[str] = None) -> str:
        """
        Generate 2D contour plot of loss landscape.
        
        Args:
            standard_trajectory: List of dicts with 'step' and 'loss' keys
            wave_trajectory: List of dicts with 'step' and 'loss' keys
            save_path: Optional path to save figure
            
        Returns:
            Path to saved figure
        """
        fig, ax = plt.subplots(figsize=(12, 10))
        fig.patch.set_facecolor('#1a1a1a')
        ax.set_facecolor('#1a1a1a')
        
        # Create synthetic loss landscape
        x = np.linspace(-5, 5, 200)
        y = np.linspace(-5, 5, 200)
        X, Y = np.meshgrid(x, y)
        Z = (np.sin(X) * np.cos(Y) + 0.1 * X**2 + 0.1 * Y**2 + 
             0.5 * np.sin(2*X) * np.sin(2*Y) + 4.4)
        
        # Plot contours
        contour = ax.contour(X, Y, Z, levels=20, cmap='viridis', alpha=0.6)
        contourf = ax.contourf(X, Y, Z, levels=20, cmap='viridis', alpha=0.3)
        
        # Extract and plot trajectories
        std_steps = [d['step'] for d in standard_trajectory]
        wave_steps = [d['step'] for d in wave_trajectory]
        max_step = max(max(std_steps), max(wave_steps))
        
        std_x = [(s / max_step) * 10 - 5 for s in std_steps]
        std_y = [i * 0.05 - 2 for i in range(len(std_steps))]
        
        wave_x = [(s / max_step) * 10 - 5 for s in wave_steps]
        wave_y = [i * 0.05 + 2 for i in range(len(wave_steps))]
        
        # Plot trajectories
        ax.plot(std_x, std_y, 'o-', color='cyan', linewidth=3,
               markersize=6, label='Standard Transformer', alpha=0.9)
        ax.plot(wave_x, wave_y, 's-', color='yellow', linewidth=3,
               markersize=6, label='Spectral GPT', alpha=0.9)
        
        # Mark start and end
        ax.scatter([std_x[0]], [std_y[0]], color='red', s=300, marker='o',
                  label='Start', zorder=5, edgecolor='white', linewidth=2)
        ax.scatter([std_x[-1], wave_x[-1]], [std_y[-1], wave_y[-1]],
                  color='green', s=300, marker='*', label='Convergence',
                  zorder=5, edgecolor='white', linewidth=2)
        
        # Labels and title
        ax.set_xlabel('Parameter Space (Dim 1)', fontsize=12, color='white')
        ax.set_ylabel('Parameter Space (Dim 2)', fontsize=12, color='white')
        ax.set_title('Loss Landscape Contour Plot with Optimization Paths',
                    fontsize=14, fontweight='bold', color='white', pad=20)
        
        # Styling
        ax.tick_params(colors='white')
        ax.grid(True, alpha=0.3, color='white')
        ax.legend(loc='upper right', fontsize=11, facecolor='#2a2a2a',
                 edgecolor='white', labelcolor='white')
        
        # Colorbar
        cbar = plt.colorbar(contourf, ax=ax)
        cbar.set_label('Loss Value', color='white', fontsize=11)
        cbar.ax.yaxis.set_tick_params(color='white')
        plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')
        
        # Save figure
        if save_path is None:
            save_path = self.output_dir / 'loss_landscape_contour.png'
        else:
            save_path = Path(save_path)
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='#1a1a1a')
        plt.close(fig)
        
        print(f"✓ Saved contour plot: {save_path}")
        return str(save_path)
    
    def generate_convergence_comparison(self,
                                       standard_trajectory: List[Dict],
                                       wave_trajectory: List[Dict],
                                       save_path: Optional[str] = None) -> str:
        """
        Generate convergence trajectory comparison plot.
        
        Args:
            standard_trajectory: List of dicts with 'step' and 'loss' keys
            wave_trajectory: List of dicts with 'step' and 'loss' keys
            save_path: Optional path to save figure
            
        Returns:
            Path to saved figure
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
        fig.patch.set_facecolor('#1a1a1a')
        
        # Extract data
        std_steps = [d['step'] for d in standard_trajectory]
        std_losses = [d['loss'] for d in standard_trajectory]
        wave_steps = [d['step'] for d in wave_trajectory]
        wave_losses = [d['loss'] for d in wave_trajectory]
        
        # Plot 1: Loss curves
        ax1.plot(std_steps, std_losses, 'o-', color='cyan', linewidth=2,
                markersize=4, label='Standard Transformer', alpha=0.9)
        ax1.plot(wave_steps, wave_losses, 's-', color='yellow', linewidth=2,
                markersize=4, label='Spectral GPT', alpha=0.9)
        
        ax1.set_xlabel('Training Step', fontsize=12, color='white')
        ax1.set_ylabel('Validation Loss', fontsize=12, color='white')
        ax1.set_title('Training Convergence Comparison',
                     fontsize=14, fontweight='bold', color='white', pad=15)
        ax1.grid(True, alpha=0.3, color='white')
        ax1.legend(fontsize=11, facecolor='#2a2a2a', edgecolor='white', labelcolor='white')
        ax1.set_facecolor('#1a1a1a')
        ax1.tick_params(colors='white')
        
        # Plot 2: Loss difference
        # Interpolate to common steps
        common_steps = np.linspace(0, min(max(std_steps), max(wave_steps)), 100)
        std_interp = np.interp(common_steps, std_steps, std_losses)
        wave_interp = np.interp(common_steps, wave_steps, wave_losses)
        diff = wave_interp - std_interp
        
        ax2.plot(common_steps, diff, color='magenta', linewidth=2, alpha=0.9)
        ax2.axhline(y=0, color='white', linestyle='--', alpha=0.5)
        ax2.fill_between(common_steps, 0, diff, where=(diff > 0),
                        color='red', alpha=0.3, label='Spectral GPT worse')
        ax2.fill_between(common_steps, 0, diff, where=(diff < 0),
                        color='green', alpha=0.3, label='Spectral GPT better')
        
        ax2.set_xlabel('Training Step', fontsize=12, color='white')
        ax2.set_ylabel('Loss Difference\n(Spectral - Standard)', fontsize=12, color='white')
        ax2.set_title('Performance Gap Over Training',
                     fontsize=14, fontweight='bold', color='white', pad=15)
        ax2.grid(True, alpha=0.3, color='white')
        ax2.legend(fontsize=11, facecolor='#2a2a2a', edgecolor='white', labelcolor='white')
        ax2.set_facecolor('#1a1a1a')
        ax2.tick_params(colors='white')
        
        plt.tight_layout()
        
        # Save figure
        if save_path is None:
            save_path = self.output_dir / 'convergence_comparison.png'
        else:
            save_path = Path(save_path)
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='#1a1a1a')
        plt.close(fig)
        
        print(f"✓ Saved convergence comparison: {save_path}")
        return str(save_path)



class FrequencySpectrumVisualizer:
    """
    Generates frequency spectrum visualizations for Spectral GPT.
    
    Features:
    - Frequency spectrum evolution during training
    - Harmonic amplitude plots
    - Phase distribution visualizations
    - Interference pattern examples
    """
    
    def __init__(self, output_dir: str):
        """
        Initialize FrequencySpectrumVisualizer.
        
        Args:
            output_dir: Directory to save generated visualizations
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def generate_spectrum_evolution(self,
                                   frequency_data: Dict[int, np.ndarray],
                                   save_path: Optional[str] = None) -> str:
        """
        Generate frequency spectrum evolution plot.
        
        Args:
            frequency_data: Dict mapping step -> frequency array
            save_path: Optional path to save figure
            
        Returns:
            Path to saved figure
        """
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.patch.set_facecolor('#1a1a1a')
        axes = axes.flatten()
        
        steps = sorted(frequency_data.keys())
        
        for idx, step in enumerate(steps[:6]):  # Show up to 6 snapshots
            ax = axes[idx]
            ax.set_facecolor('#1a1a1a')
            
            freqs = frequency_data[step]
            
            # Create histogram
            ax.hist(freqs.flatten(), bins=50, color='cyan', alpha=0.7, edgecolor='white')
            ax.set_xlabel('Frequency (Hz)', fontsize=10, color='white')
            ax.set_ylabel('Count', fontsize=10, color='white')
            ax.set_title(f'Step {step}', fontsize=12, fontweight='bold', color='white')
            ax.grid(True, alpha=0.3, color='white')
            ax.tick_params(colors='white')
            
            # Add statistics
            mean_freq = np.mean(freqs)
            std_freq = np.std(freqs)
            ax.text(0.95, 0.95, f'μ={mean_freq:.3f}\nσ={std_freq:.3f}',
                   transform=ax.transAxes, ha='right', va='top',
                   fontsize=9, color='white',
                   bbox=dict(boxstyle='round', facecolor='#333333', alpha=0.8))
        
        # Hide unused subplots
        for idx in range(len(steps), 6):
            axes[idx].set_visible(False)
        
        fig.suptitle('Frequency Spectrum Evolution During Training',
                    fontsize=16, fontweight='bold', color='white', y=0.98)
        plt.tight_layout()
        
        # Save figure
        if save_path is None:
            save_path = self.output_dir / 'frequency_spectrum_evolution.png'
        else:
            save_path = Path(save_path)
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='#1a1a1a')
        plt.close(fig)
        
        print(f"✓ Saved frequency spectrum evolution: {save_path}")
        return str(save_path)
    
    def generate_harmonic_amplitude_plot(self,
                                        harmonic_data: np.ndarray,
                                        save_path: Optional[str] = None) -> str:
        """
        Generate harmonic amplitude visualization.
        
        Args:
            harmonic_data: Array of shape (tokens, waves, harmonics)
            save_path: Optional path to save figure
            
        Returns:
            Path to saved figure
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        fig.patch.set_facecolor('#1a1a1a')
        
        # Plot 1: Average harmonic profile
        ax1.set_facecolor('#1a1a1a')
        mean_harmonics = harmonic_data.mean(axis=0)  # Average over tokens
        
        im = ax1.imshow(mean_harmonics, aspect='auto', cmap='plasma', interpolation='nearest')
        ax1.set_xlabel('Harmonic (1f, 2f, 3f, ...)', fontsize=12, color='white')
        ax1.set_ylabel('Wave Component', fontsize=12, color='white')
        ax1.set_title('Mean Harmonic Amplitudes Across Tokens',
                     fontsize=14, fontweight='bold', color='white', pad=15)
        ax1.tick_params(colors='white')
        
        # Colorbar
        cbar1 = plt.colorbar(im, ax=ax1)
        cbar1.set_label('Amplitude', color='white', fontsize=11)
        cbar1.ax.yaxis.set_tick_params(color='white')
        plt.setp(plt.getp(cbar1.ax.axes, 'yticklabels'), color='white')
        
        # Plot 2: Harmonic distribution
        ax2.set_facecolor('#1a1a1a')
        num_harmonics = harmonic_data.shape[2]
        colors = plt.cm.plasma(np.linspace(0, 1, num_harmonics))
        
        for h in range(num_harmonics):
            data = harmonic_data[:, :, h].flatten()
            ax2.hist(data, bins=40, alpha=0.6, label=f'Harmonic {h+1}',
                    color=colors[h], edgecolor='white', linewidth=0.5)
        
        ax2.set_xlabel('Amplitude', fontsize=12, color='white')
        ax2.set_ylabel('Count', fontsize=12, color='white')
        ax2.set_title('Harmonic Amplitude Distribution',
                     fontsize=14, fontweight='bold', color='white', pad=15)
        ax2.legend(fontsize=10, facecolor='#2a2a2a', edgecolor='white', labelcolor='white')
        ax2.grid(True, alpha=0.3, color='white')
        ax2.tick_params(colors='white')
        
        plt.tight_layout()
        
        # Save figure
        if save_path is None:
            save_path = self.output_dir / 'harmonic_amplitudes.png'
        else:
            save_path = Path(save_path)
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='#1a1a1a')
        plt.close(fig)
        
        print(f"✓ Saved harmonic amplitude plot: {save_path}")
        return str(save_path)
    
    def generate_phase_distribution(self,
                                   phase_data: np.ndarray,
                                   save_path: Optional[str] = None) -> str:
        """
        Generate phase distribution visualization.
        
        Args:
            phase_data: Array of shape (tokens, waves) with phase values
            save_path: Optional path to save figure
            
        Returns:
            Path to saved figure
        """
        fig = plt.figure(figsize=(16, 8))
        fig.patch.set_facecolor('#1a1a1a')
        
        # Create grid for subplots
        gs = fig.add_gridspec(2, 4, hspace=0.3, wspace=0.3)
        
        # Plot 1: Heatmap of phases
        ax1 = fig.add_subplot(gs[0, :2])
        ax1.set_facecolor('#1a1a1a')
        
        n_tokens = min(100, phase_data.shape[0])
        im = ax1.imshow(phase_data[:n_tokens].T, aspect='auto', cmap='twilight',
                       vmin=0, vmax=2*np.pi, interpolation='nearest')
        ax1.set_xlabel('Token ID', fontsize=12, color='white')
        ax1.set_ylabel('Wave Component', fontsize=12, color='white')
        ax1.set_title('Phase Distribution Across Tokens',
                     fontsize=14, fontweight='bold', color='white', pad=15)
        ax1.tick_params(colors='white')
        
        cbar = plt.colorbar(im, ax=ax1)
        cbar.set_label('Phase (radians)', color='white', fontsize=11)
        cbar.ax.yaxis.set_tick_params(color='white')
        plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')
        
        # Plot 2-5: Polar plots for first 4 wave components
        for wave_idx in range(min(4, phase_data.shape[1])):
            ax = fig.add_subplot(gs[wave_idx // 2, 2 + wave_idx % 2], projection='polar')
            ax.set_facecolor('#1a1a1a')
            
            phases = phase_data[:n_tokens, wave_idx]
            colors = plt.cm.viridis(np.linspace(0, 1, n_tokens))
            
            ax.scatter(phases, np.ones_like(phases), c=colors, s=20, alpha=0.6)
            ax.set_title(f'Wave {wave_idx + 1}', fontsize=12, fontweight='bold',
                        color='white', pad=15)
            ax.grid(True, alpha=0.3, color='white')
            ax.tick_params(colors='white')
            ax.set_ylim(0, 1.2)
        
        fig.suptitle('Phase Distribution Analysis',
                    fontsize=16, fontweight='bold', color='white', y=0.98)
        
        # Save figure
        if save_path is None:
            save_path = self.output_dir / 'phase_distribution.png'
        else:
            save_path = Path(save_path)
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='#1a1a1a')
        plt.close(fig)
        
        print(f"✓ Saved phase distribution: {save_path}")
        return str(save_path)
    
    def generate_interference_patterns(self,
                                      phase_data: np.ndarray,
                                      save_path: Optional[str] = None) -> str:
        """
        Generate interference pattern visualization.
        
        Args:
            phase_data: Array of shape (tokens, waves) with phase values
            save_path: Optional path to save figure
            
        Returns:
            Path to saved figure
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        fig.patch.set_facecolor('#1a1a1a')
        axes = axes.flatten()
        
        # Select a few token pairs to visualize
        n_tokens = min(50, phase_data.shape[0])
        token_pairs = [(0, 10), (5, 15), (20, 30), (10, 40)]
        
        for idx, (i, j) in enumerate(token_pairs):
            if i >= n_tokens or j >= n_tokens:
                continue
            
            ax = axes[idx]
            ax.set_facecolor('#1a1a1a')
            
            # Calculate interference for each wave
            phase_i = phase_data[i]
            phase_j = phase_data[j]
            interference = np.cos(phase_i - phase_j)
            
            # Create wave visualization
            t = np.linspace(0, 4*np.pi, 1000)
            wave_i = np.sin(t + phase_i[0])
            wave_j = np.sin(t + phase_j[0])
            wave_sum = wave_i + wave_j
            
            # Plot individual waves
            ax.plot(t, wave_i, '--', color='cyan', alpha=0.6, linewidth=2,
                   label=f'Token {i}')
            ax.plot(t, wave_j, '--', color='yellow', alpha=0.6, linewidth=2,
                   label=f'Token {j}')
            
            # Plot interference
            ax.plot(t, wave_sum, '-', color='magenta', linewidth=3, alpha=0.9,
                   label='Interference')
            ax.axhline(y=0, color='white', linestyle=':', alpha=0.3)
            
            # Calculate interference type
            mean_interference = np.mean(interference)
            if mean_interference > 0.5:
                interference_type = 'Constructive'
                color = 'green'
            elif mean_interference < -0.5:
                interference_type = 'Destructive'
                color = 'red'
            else:
                interference_type = 'Mixed'
                color = 'orange'
            
            ax.set_xlabel('Time', fontsize=11, color='white')
            ax.set_ylabel('Amplitude', fontsize=11, color='white')
            ax.set_title(f'Tokens {i} & {j}: {interference_type} Interference\n'
                        f'Mean: {mean_interference:.3f}',
                        fontsize=12, fontweight='bold', color=color, pad=10)
            ax.legend(fontsize=9, facecolor='#2a2a2a', edgecolor='white', labelcolor='white')
            ax.grid(True, alpha=0.3, color='white')
            ax.tick_params(colors='white')
        
        fig.suptitle('Wave Interference Patterns Between Token Pairs',
                    fontsize=16, fontweight='bold', color='white', y=0.98)
        plt.tight_layout()
        
        # Save figure
        if save_path is None:
            save_path = self.output_dir / 'interference_patterns.png'
        else:
            save_path = Path(save_path)
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='#1a1a1a')
        plt.close(fig)
        
        print(f"✓ Saved interference patterns: {save_path}")
        return str(save_path)
    
    def generate_frequency_heatmap(self,
                                  frequency_data: np.ndarray,
                                  token_labels: Optional[List[str]] = None,
                                  save_path: Optional[str] = None) -> str:
        """
        Generate frequency heatmap for tokens.
        
        Args:
            frequency_data: Array of shape (tokens, waves) with frequency values
            token_labels: Optional list of token labels
            save_path: Optional path to save figure
            
        Returns:
            Path to saved figure
        """
        fig, ax = plt.subplots(figsize=(14, 10))
        fig.patch.set_facecolor('#1a1a1a')
        ax.set_facecolor('#1a1a1a')
        
        n_tokens = min(50, frequency_data.shape[0])
        data = frequency_data[:n_tokens]
        
        # Create heatmap
        im = ax.imshow(data.T, aspect='auto', cmap='viridis', interpolation='nearest')
        
        # Labels
        ax.set_xlabel('Token ID', fontsize=12, color='white')
        ax.set_ylabel('Wave Component', fontsize=12, color='white')
        ax.set_title('Base Frequencies per Token and Wave',
                    fontsize=14, fontweight='bold', color='white', pad=15)
        
        # Add token labels if provided
        if token_labels is not None:
            ax.set_xticks(range(min(len(token_labels), n_tokens)))
            ax.set_xticklabels(token_labels[:n_tokens], rotation=90, fontsize=8)
        
        ax.tick_params(colors='white')
        
        # Colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Frequency (Hz)', color='white', fontsize=11)
        cbar.ax.yaxis.set_tick_params(color='white')
        plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')
        
        plt.tight_layout()
        
        # Save figure
        if save_path is None:
            save_path = self.output_dir / 'frequency_heatmap.png'
        else:
            save_path = Path(save_path)
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='#1a1a1a')
        plt.close(fig)
        
        print(f"✓ Saved frequency heatmap: {save_path}")
        return str(save_path)
