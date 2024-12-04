"""
NeuralDataVisualizer: A comprehensive tool for visualizing neural activity data.

This module provides an easy-to-use interface for creating publication-quality
visualizations of neural activity data. It supports multiple visualization types,
customizable styling, and various data transformations.

Features:
- Multiple visualization types (individual channels, average activity, etc.)
- Customizable styling and themes
- Support for various units and transformations
- Publication-ready output
- Interactive plot options
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import Dict, Optional, Tuple, List, Union
from dataclasses import dataclass
import colorsys
import scipy.cluster.hierarchy as sch
from matplotlib import gridspec

@dataclass
class PlotConfig:
    """Configuration class for plot styling."""
    figsize: Tuple[int, int] = (15, 8)
    dpi: int = 100
    title_fontsize: int = 16
    label_fontsize: int = 12
    tick_fontsize: int = 10
    line_width: float = 1.2
    grid: bool = True
    grid_style: str = '--'
    grid_alpha: float = 0.7
    legend_position: str = 'right'  # 'right', 'bottom', or 'inside'

class NeuralDataVisualizer:
    """
    A class for visualizing neural activity data with customizable plotting options.
    
    Attributes:
        style (str): The seaborn style to use ('whitegrid', 'darkgrid', etc.)
        context (str): The plotting context ('notebook', 'talk', 'paper', 'poster')
        color_scheme (str): Color scheme for plots ('husl', 'rainbow', etc.)
        n_colors (int): Number of colors in the palette
    """
    
    def __init__(self, 
                 style: str = 'whitegrid', 
                 context: str = 'notebook',
                 color_scheme: str = 'husl',
                 n_colors: int = 20):
        """
        Initialize the visualizer with custom styling options.
        
        Args:
            style: Seaborn style name
            context: Seaborn context name
            color_scheme: Color palette name
            n_colors: Number of colors in the palette
        """
        self.set_style(style, context)
        self.set_color_palette(color_scheme, n_colors)
        self.plot_config = PlotConfig()

    def set_color_palette(self, color_scheme, n_colors):
        self.color_palette = self._generate_palette(color_scheme, n_colors)

    def _generate_palette(self, scheme: str, n_colors: int) -> List:
        """Generate a color palette with improved contrast."""
        if scheme == 'rainbow':
            return [colorsys.hsv_to_rgb(i/n_colors, 0.8, 0.9) for i in range(n_colors)]
        return sns.color_palette(scheme, n_colors)
    
    def set_style(self, style: str, context: str):
        """Configure the plotting style and context."""
        sns.set_style(style)
        sns.set_context(context)
        plt.rcParams.update({
            'font.family': 'sans-serif',
            'font.sans-serif': ['Arial'],
            'axes.titleweight': 'bold',
            'axes.spines.top': False,
            'axes.spines.right': False
        })
    
    def _setup_legend(self, ax: plt.Axes, config: PlotConfig):
        """Configure legend position and style."""
        if config.legend_position == 'right':
            ax.legend(bbox_to_anchor=(1.05, 1), 
                     loc='upper left',
                     borderaxespad=0.,
                     fontsize=config.label_fontsize-2)
        elif config.legend_position == 'bottom':
            ax.legend(bbox_to_anchor=(0.5, -0.15),
                     loc='upper center',
                     borderaxespad=0.,
                     fontsize=config.label_fontsize-2,
                     ncol=3)
        else:  # 'inside'
            ax.legend(loc='best', fontsize=config.label_fontsize-2)
    
    def plot_channels(self, 
                     data: Dict,
                     time_period_name: str,
                     window: int,
                     config: Optional[PlotConfig] = None,
                     conv_to_uv: bool = True,
                     ylim: Optional[tuple] = None,
                     highlight_channels: Optional[List[str]] = None,
                     show_events: Optional[Dict[str, float]] = None) -> Tuple[plt.Figure, plt.Axes]:
        """
        Plot individual channel data with enhanced visualization options.
        
        Args:
            data: Dictionary containing time and channel data
            time_period_name: Name of the time period for the title
            config: PlotConfig object for customizing the plot
            conv_to_uv: Convert values to microvolts
            ylim: Y-axis limits (min, max)
            highlight_channels: List of channel names to highlight
            show_events: Dictionary of event names and their timestamps
            
        Returns:
            Tuple of (Figure, Axes) objects
        """
        config = config or self.plot_config
        fig, ax = plt.subplots(figsize=config.figsize, dpi=config.dpi)
        
        # Plot channels
        for idx, (channel_name, channel_data) in enumerate(data.items()):
            if channel_name == 'time':
                continue
                
            is_highlighted = highlight_channels and channel_name in highlight_channels
            line_width = config.line_width * (1.5 if is_highlighted else 1.0)
            alpha = 1.0 if is_highlighted else 0.7
            
            ax.plot(data['time']/1000, 
                   channel_data, 
                   label=f'Channel {channel_name}',
                   color=self.color_palette[idx % len(self.color_palette)],
                   linewidth=line_width,
                   alpha=alpha)
        
        # Add events if specified
        if show_events:
            for event_name, timestamp in show_events.items():
                ax.axvline(x=timestamp/1000, color='red', linestyle='--', alpha=0.5)
                ax.text(timestamp/1000, ax.get_ylim()[1], event_name, 
                       rotation=90, va='bottom')
        
        # Customize appearance
        ax.set_title(f'Time Period of Neural Activity: from {time_period_name}s to {int(time_period_name)+window}s', 
                    fontsize=config.title_fontsize, 
                    pad=20)
        
        ax.set_xlabel('Time (s)', fontsize=config.label_fontsize)
        ax.set_ylabel('Voltage (µV)' if conv_to_uv else 'Raw value', 
                     fontsize=config.label_fontsize)
        
        ax.tick_params(axis='both', labelsize=config.tick_fontsize)
        
        if ylim:
            ax.set_ylim(ylim)
            
        if config.grid:
            ax.grid(True, linestyle=config.grid_style, alpha=config.grid_alpha)
            
        self._setup_legend(ax, config)
        plt.tight_layout()
        
        return fig, ax
    
    def plot_average_activity(self,
                            data: Dict,
                            time_period_name: str,
                            config: Optional[PlotConfig] = None,
                            conv_to_uv: bool = True,
                            show_std: bool = True,
                            show_percentiles: bool = False,
                            smooth_window: Optional[int] = None) -> Tuple[plt.Figure, plt.Axes]:
        """
        Plot average neural activity with enhanced visualization options.
        
        Args:
            data: Dictionary containing time and channel data
            time_period_name: Name of the time period for the title
            config: PlotConfig object for customizing the plot
            conv_to_uv: Convert values to microvolts
            show_std: Show standard deviation
            show_percentiles: Show 25th and 75th percentiles
            smooth_window: Window size for moving average smoothing
            
        Returns:
            Tuple of (Figure, Axes) objects
        """
        config = config or self.plot_config
        
        channel_data = [data[ch] for ch in data.keys() if ch != 'time']
        mean_activity = np.mean(channel_data, axis=0)
        
        if smooth_window:
            kernel = np.ones(smooth_window) / smooth_window
            mean_activity = np.convolve(mean_activity, kernel, mode='same')
        
        fig, ax = plt.subplots(figsize=config.figsize, dpi=config.dpi)
        
        # Plot mean activity
        ax.plot(data['time']/1000, mean_activity, 
                color='blue', linewidth=config.line_width*1.2, 
                label='Mean activity')
        
        if show_std:
            std_activity = np.std(channel_data, axis=0)
            ax.fill_between(data['time']/1000, 
                          mean_activity - std_activity,
                          mean_activity + std_activity,
                          alpha=0.2, color='blue',
                          label='±1 SD')
        
        if show_percentiles:
            percentile_25 = np.percentile(channel_data, 25, axis=0)
            percentile_75 = np.percentile(channel_data, 75, axis=0)
            ax.fill_between(data['time']/1000, 
                          percentile_25,
                          percentile_75,
                          alpha=0.1, color='blue',
                          label='25-75th percentile')
        
        # Customize appearance
        ax.set_title(f'Average Neural Activity - {time_period_name}',
                    fontsize=config.title_fontsize, pad=20)
        ax.set_xlabel('Time (ms)', fontsize=config.label_fontsize)
        ax.set_ylabel('Voltage (µV)' if conv_to_uv else 'Raw value',
                     fontsize=config.label_fontsize)
        
        ax.tick_params(axis='both', labelsize=config.tick_fontsize)
        
        if config.grid:
            ax.grid(True, linestyle=config.grid_style, alpha=config.grid_alpha)
            
        self._setup_legend(ax, config)
        plt.tight_layout()
        
        return fig, ax
    
    def plot_sorted_correlation_matrix(self, correlation_matrix, linkage_matrix):
        fig = plt.figure(figsize=(5, 2), dpi=150) 
        gs = gridspec.GridSpec(1, 2, width_ratios=[1, 8]) 
        ax0 = plt.subplot(gs[0])
        ax1 = plt.subplot(gs[1])

        sch.set_link_color_palette(['orange', 'lightblue'])
        with plt.rc_context({'lines.linewidth': 0.7}):
            dendrogram = sch.dendrogram(linkage_matrix, ax=ax0, no_labels=True, orientation='left', above_threshold_color='grey')

        ax0.tick_params(axis='x', which='major', labelsize=6)
        ax0.grid(False)
        ax0.set_xlabel('Distance', fontsize=10)
        ax0.spines['left'].set_edgecolor("white")

        sorted_indices = list(map(int, dendrogram['ivl'][::-1])) 
        sorted_correlation_matrix = correlation_matrix[sorted_indices][:, sorted_indices]

        sns.heatmap(sorted_correlation_matrix, cmap='viridis',  cbar_kws={"shrink": 0.5, 'label': 'correlation'}, vmin=-1, vmax=1, ax=ax1)
    
        ax1.tick_params(axis='x', which='major', labelsize=8)
        ax1.tick_params(axis='y', which='major', labelsize=8)

        plt.title('Sorted Correlation Matrix')
        ax1.set_xlabel('Sample Index', fontsize=10)

        plt.grid(False)
        plt.show()

        return fig, ax0, ax1
    
    def plot_transition_frequency(self, x, y1, y2, intersection_point):
        fig = plt.figure(figsize=(5, 2))
        plt.plot(x, y1, label='Cluster 1', color='blue')
        plt.plot(x, y2, label='Cluster 2', color='red')
        plt.xlabel('Frequency', fontsize=10)
        plt.ylabel('Power', fontsize=10)
        plt.title('Average Normalized PSD for Clusters') 
        plt.legend(fontsize=10)
        plt.tick_params(axis='x', which='major', labelsize=8)
        plt.tick_params(axis='y', which='major', labelsize=8)
        plt.grid(True)

        plt.xlim(0, 35)

        for (x_inter,y_iter) in intersection_point:
            plt.axvline(x=x_inter, color='green', linestyle='--')
            plt.text(x_inter + 0.5, y_iter + 0.1, f'{x_inter:.2f}', fontsize=8) 
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)

        plt.show()

        return fig
    
    def plot_delta_beta_matrix(self, data, num_bins_x = 6):
        num_y = len(data) // num_bins_x
        delta_beta_matrix = data[:num_bins_x*num_y].reshape((num_y, num_bins_x))

        fig = plt.figure(figsize=(5, 2))
        plt.imshow(delta_beta_matrix, aspect='auto', cmap='viridis', interpolation='nearest')
        plt.colorbar(label='Delta/Beta Power Ratio')
        plt.title('Delta/Beta Power Ratio for 30-Minute Intervals', fontsize=14)
        # plt.xlabel('Time [Minutes]', fontsize=16)
        # plt.ylabel('Half-Hour count', fontsize=16)
        # plt.xticks(tick_locations, tick_labels, fontsize=14)
        # plt.yticks(fontsize=14)
        plt.grid(False)
        plt.show()

        return fig
    
    def plot_autocorr(self, autocorr, avg_peak_index, peak_range = 300, bin_second = 10):
        start_index = max(avg_peak_index - (peak_range // bin_second), 0) 
        end_index = min(avg_peak_index + (peak_range // bin_second), len(autocorr))

        autocorr_plot = autocorr[start_index:end_index + 1]
        avg_autocorr_range = np.mean(autocorr_plot)
        lags = np.arange(-len(autocorr_plot)//2 + 1, len(autocorr_plot)//2 + 1)* 10 

        fig = plt.figure(figsize=(5, 2))
        plt.plot(lags, autocorr_plot, color='red', label='Autocorrelation')
        plt.axhline(y=avg_autocorr_range, color='black', linestyle='--', label='Average Autocorrelation')
        plt.xlabel('Lag [s]',fontsize=14)
        plt.ylabel('Autocorr.',fontsize=14)
        plt.title(f'Average Autocorrelation Function within {peak_range} seconds around the peak', fontsize=14)
        plt.legend(fontsize=14)
        plt.grid(True)
        plt.show()

        return fig
    
    def plot_kde_distribution(self, x_values, y_values, avg_dist_p2p, lower_bound, upper_bound, std_dev):
        # Plot the KDE Distribution
        plt.figure(figsize=(5, 2))
        plt.plot(x_values, y_values, color='skyblue', label='Sleep Time Distribution (KDE)')

        # Add Average and Confidence Interval
        plt.axvline(avg_dist_p2p, color='red', linestyle='dashed',
                    linewidth=1, label=f'Average ({avg_dist_p2p:.2f} seconds)')
        plt.axvspan(lower_bound, upper_bound, color='gray', alpha=0.3, label='95% Confidence Interval')

        # Plot Labels and Title
        plt.xlabel('Cycle Length [seconds]', fontsize=14)
        plt.ylabel('Density', fontsize=14)
        plt.title('Distribution of Sleep Cycle Times', fontsize=14)
        plt.legend()
        plt.grid(axis='y', alpha=0.5)  # Optional: Add gridlines for clarity
        plt.legend(fontsize=12)
        plt.xticks(fontsize=12)  # Larger tick labels for x-axis
        plt.yticks(fontsize=12)

        # Add Standard Deviation
        plt.text(0.75, 0.8, f'Std. Dev. = {std_dev:.2f} seconds', transform=plt.gca().transAxes, fontsize=10)
        plt.show()


# Example usage
if __name__ == "__main__":
    # Generate sample data
    time = np.linspace(0, 1, 1000)
    channels = {
        'time': time,
        'ch1': np.sin(2 * np.pi * 5 * time) + np.random.normal(0, 0.1, len(time)),
        'ch2': np.sin(2 * np.pi * 7 * time) + np.random.normal(0, 0.1, len(time)),
        'ch3': np.sin(2 * np.pi * 3 * time) + np.random.normal(0, 0.1, len(time))
    }
    
    # Create visualizer instance
    viz = NeuralDataVisualizer(style='whitegrid', context='talk')
    
    # Custom configuration
    custom_config = PlotConfig(
        figsize=(12, 6),
        dpi=120,
        title_fontsize=18,
        label_fontsize=14,
        legend_position='right'
    )
    
    # Plot channels with events
    events = {'Stimulus': 0.3, 'Response': 0.7}
    fig1, ax1 = viz.plot_channels(channels, 'Sample Period',
                                config=custom_config,
                                highlight_channels=['ch1'],
                                show_events=events)
    
    # Plot average activity with percentiles
    fig2, ax2 = viz.plot_average_activity(channels, 'Sample Period',
                                        config=custom_config,
                                        show_std=True,
                                        show_percentiles=True,
                                        smooth_window=5)
    
    plt.show()