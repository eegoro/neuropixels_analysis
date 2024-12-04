# imports
import numpy as np
from scipy.signal import butter, filtfilt, resample, welch, find_peaks
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.stats import pearsonr, zscore, gaussian_kde
import numpy as np
import matplotlib.pyplot as plt
import pickle
from tqdm import tqdm
import os
import scipy.stats as st
from typing import Dict, List, Tuple, Optional, Generator, Union
from visualization import visualizer
from collections import OrderedDict

class NeuropixelsDataProcessor:

    def __init__(
        self, 
        sampling_rate: float, 
        downsampled_rate: float = 250, 
        low_pass_cutoff: int = 100,
        batch_size = 1024**2,
        overlap: int = 1024,
        bin_second: int = 10
    ):
        """
        Initialize the data processor with sampling parameters.
        
        Args:
            sampling_rate: Original data sampling rate (Hz)
            target_sample_rate: Desired output sampling rate (Hz)
            batch_size: Maximum batch size for processing
        """
        self.sampling_rate = sampling_rate
        self.batch_size = batch_size
        self.overlap = overlap

        self.nyquist_frequency = 0.5 * self.sampling_rate

        self.low_pass_cutoff = low_pass_cutoff  # Hz
        self.downsampled_rate = downsampled_rate  # Hz

        self.bin_second = bin_second  # seconds
        self.bin_size = self.bin_second * self.downsampled_rate

        self.delta_window = (0, 3)
        self.beta_window = (4, 30)

    def create_batch(self,
        data: np.ndarray, 
        batch_size: int,
        overlap = 0
        ):

        for start in range(0, len(data), batch_size):
            end = min(start + batch_size, len(data))
            size_without_overlap = end-start
            start = max(0, start-overlap)
            yield (size_without_overlap, data[start:end])


    def preprocess_signal(self, data, batch_size, overlap):
        """
        Preprocess the data by:
        1. Low-pass filtering
        2. Downsampling
        """

        order = 8
        b, a = butter(order, self.low_pass_cutoff / self.nyquist_frequency, btype='low')
        preprocess_data = []

        for (size, batch) in self.create_batch(data, batch_size = batch_size, overlap=overlap):
            filtered_batch = filtfilt(b, a, batch)[-size:]
            downsampled_batch = resample(filtered_batch,
                                         int(batch_size * self.downsampled_rate / self.sampling_rate))
            preprocess_data.append(downsampled_batch)

        return np.concatenate(preprocess_data)
    
    def compute_power_spectrum(self, data: np.ndarray, bin_size):
        power_spectrum = []
        
        for (_, bin_batch) in self.create_batch(data, batch_size = bin_size, overlap=0):
            frequencies, Pxx = welch(
                bin_batch, 
                fs=self.downsampled_rate, 
                nperseg=self.downsampled_rate, # 1 second window
                noverlap=self.downsampled_rate // 2 # 50% overlap
            )
            power_spectrum.append(Pxx)
        
        return frequencies, np.array(power_spectrum) 
    
    def normalize_power_spectrum(self, data: np.ndarray) -> np.ndarray:
        avg_psd = np.mean(data, axis=0)
        std_per_f = np.std(data, axis=0)
        return (data - avg_psd) / std_per_f

    
    def cluster_correlation_matrix(self, correlation_matrix, num_clusters=20):
        linkage_matrix = linkage(correlation_matrix, method='ward') # hierarchical clustering
        cluster_labels = fcluster(linkage_matrix, num_clusters, criterion='maxclust')  # Cut the dendrogram to get clusters
       
        clusters = {}
        for i, label in enumerate(cluster_labels):
            clusters.setdefault(label, []).append(i)

        return linkage_matrix, cluster_labels, clusters
    
    def calculate_transition_frequency(self, x, y1, y2):
        diff = y1 - y2
        idx = np.where(np.diff(np.sign(diff)))[0]
        x_intersections = x[idx] - diff[idx] * (x[idx + 1] - x[idx]) / (diff[idx + 1] - diff[idx])
        y_intersections = np.interp(x_intersections, x, y1)

        return np.stack([x_intersections,y_intersections], axis=1)

    def calculate_delta_beta_ratio(self, power_spectrum: np.ndarray):
        """
        Calculate delta/beta power ratio
        
        Parameters:
        -----------
        power_spectrum : np.ndarray
            Power spectrum
        
        Returns:
        --------
        float
            Delta/Beta ratio
        """
        power_spectrum_delta = power_spectrum[:, self.delta_window[0]:self.delta_window[1]]
        power_spectrum_beta = power_spectrum[:, self.beta_window[0]:self.beta_window[1]]
        delta_mean_power = np.mean(power_spectrum_delta, axis=1)
        beta_mean_power = np.mean(power_spectrum_beta, axis=1)
        delta_beta_ratio = delta_mean_power / beta_mean_power

        return delta_beta_ratio
    
    def calculate_autocorrelation(self, data):
        autocorr = np.correlate(data, data, mode='full')
        norm_autocorr = autocorr / np.max(autocorr)
        avg_peak_index = np.argmax(autocorr)
        
        return autocorr, norm_autocorr, avg_peak_index
    
    def analyze_sleep_cycles(self, norm_autocorr, bin_second = 10, confidence_level = 0.95, smooth = 500):

        peaks, _ = find_peaks(norm_autocorr)
        peak_distances = np.diff(peaks)

        avg_dist_p2p = np.mean(peak_distances) * bin_second
        
        variance = np.var(peak_distances) * bin_second
        std_dev = np.std(peak_distances) * bin_second

        z_score = st.norm.ppf((1 + confidence_level) / 2)
        margin_of_error = z_score * std_dev / np.sqrt(len(peak_distances))
        lower_bound = avg_dist_p2p - margin_of_error
        upper_bound = avg_dist_p2p + margin_of_error
        z_scores = zscore(peak_distances)

        time_values = peak_distances * bin_second 
        kde = gaussian_kde(time_values)

        x_values = np.linspace(time_values.min(), time_values.max(), smooth)
        y_values = kde(x_values)

        return {'average distance between cycles': avg_dist_p2p,
                'standard deviation': std_dev,
                '95 percent confidence interval': (lower_bound, upper_bound),
                'maximal cycle time (sec)': np.max(time_values),
                'minimal cycle time (sec)': np.min(time_values),
                'kde': (x_values, y_values)
                }
        
    def pipeline(self, data):
        viz = visualizer.NeuralDataVisualizer(style='whitegrid', context='paper', color_scheme = "Paired")

        resample_data = self.preprocess_signal(data, self.batch_size, self.overlap)
        freq, PSD = self.compute_power_spectrum(resample_data, self.bin_size)
        normalized_PSD = self.normalize_power_spectrum(PSD)

        nPSD_low = normalized_PSD[:,:30]

        correlation_matrix = np.corrcoef(nPSD_low)
        linkage_matrix, cluster_labels, clusters = self.cluster_correlation_matrix(correlation_matrix, num_clusters=20)

        viz.plot_sorted_correlation_matrix(correlation_matrix, linkage_matrix)
        
        avg_psd_cluster = []
        clusters = OrderedDict(sorted(clusters.items(), reverse=True, key = lambda x : len(x[1])))
        for indexes in clusters.values():
            avg_psd_cluster.append(np.mean(nPSD_low[indexes], axis=0))

        transition_frequencies = self.calculate_transition_frequency(freq[:30], avg_psd_cluster[0], avg_psd_cluster[1])
        
        viz.plot_transition_frequency(freq[:30], avg_psd_cluster[0], avg_psd_cluster[1], transition_frequencies)

        delta_beta_ratio = self.calculate_delta_beta_ratio(PSD)

        viz.plot_delta_beta_matrix(delta_beta_ratio, num_bins_x = 6)

        autocorr, norm_autocorr, avg_peak_index = self.calculate_autocorrelation(delta_beta_ratio)

        viz.plot_autocorr(norm_autocorr, avg_peak_index, peak_range = 300, bin_second = self.bin_second)

        dict_sleep_cycles = self.analyze_sleep_cycles(norm_autocorr)

        print("Sleep Cycle duration analysis")
        for key, value in dict_sleep_cycles.items():
            if key=='kde':
                continue
            print("{0}: {1}".format(key,value))

        x_values, y_values = dict_sleep_cycles['kde']
        avg_dist_p2p = dict_sleep_cycles['average distance between cycles']
        lower_bound, upper_bound = dict_sleep_cycles['95 percent confidence interval']
        std_dev = dict_sleep_cycles['standard deviation']

        viz.plot_kde_distribution(x_values, y_values, avg_dist_p2p, lower_bound, upper_bound, std_dev)

        return None