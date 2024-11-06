"""
Utilities for reading Neuropixels neural recording data

This module provides tools for reading and processing data from Neuropixels probes.
Supports Neuropixels 2.0 and others variants.

"""

import os
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Union

class NeuropixelsReader:
    """
    A class to read and process Neuropixels neural recording data.

    Handles data from Neuropixels probe with appropriate
    gain corrections and channel mapping.

    """

    def __init__(self, file_path: Union[str, Path]):
        """
        Initialize the NeuropixelsReader with a binary file path.

        Args:
            file_path: Path to the .bin recording file
        """
        self.file_path = Path(file_path)

        if not self.file_path.exists():
            raise FileNotFoundError(f"Recording file not found: {self.file_path}")
        
        self.metadata = self._read_metadata()

        self.is_imec = self.metadata.get('typeThis', '') == 'imec'
        self.sampling_rate = float(self.metadata.get(
            'imSampRate' if self.is_imec else 'niSampRate', 0))
        
        # Get channel counts
        counts = list(map(int, self.metadata.get('snsApLfSy', '0,0,0').split(',')))
        self.n_ap_channels = counts[0]
        self.n_lf_channels = counts[1]
        self.n_sync_channels = counts[2]
        self.n_total_channels = int(self.metadata.get('nSavedChans', 0))

        # Set voltage conversion parameters
        self.max_int = int(self.metadata.get('imMaxInt', 512)) if self.is_imec else 32768
        self.voltage_range = float(self.metadata.get(
            'imAiRangeMax' if self.is_imec else 'niAiRangeMax', 0))
        
        # Set gains based on probe type
        self.probe_type = int(self.metadata.get('imDatPrb_type', 0))
        self.ap_gains, self.lf_gains = self._get_channel_gains()

        # Create channel mapping
        self.channel_map = self._create_channel_map()

    def _read_metadata(self) -> Dict[str, str]:
        """
        Read metadata from the corresponding .meta file.

        Returns:
            Dict containing parsed metadata
        """
        meta_path = self.file_path.with_suffix('.meta')

        if not meta_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {meta_path}")
        
        metadata = {}
        try:
            with open(meta_path, 'r') as file:
                for line in file:
                    if '=' in line:
                        key, value = line.strip().split('=', 1)
                        metadata[key.lstrip('~')] = value
        except IOError as e:
            raise IOError(f"Could not read metadata file: {meta_path}") from e
        
        return metadata
    
    def _get_channel_gains(self) -> Tuple[List[float], List[float]]:
        """Get gain values for AP and LF channels."""
        # For Neuropixels 2.0 (probe types 21, 24)
        if self.probe_type in [21, 24]:
            return [80.0] * self.n_ap_channels, [80.0] * self.n_lf_channels
            
        # For Neuropixels 3A/B
        imro_table = self.metadata.get('imroTbl', '')
        if not imro_table:
            return [], []
            
        ap_gains = []
        lf_gains = []
        
        # Parse IMRO table
        lines = imro_table.replace(')(', ')|(').split('|')[1:]
        for line in lines:
            if not line.strip():
                continue
                
            parts = line.strip('()').split()
            if len(parts) >= 4:
                ap_gains.append(float(parts[3]))
                lf_gains.append(float(parts[4]) if len(parts) > 4 else 0.0)
                
        return ap_gains, lf_gains
    
    def _create_channel_map(self) -> Dict[int, int]:
        """Create mapping between logical and physical channel indices."""
        channel_subset = self.metadata.get('snsSaveChanSubset', 'all')
        
        if channel_subset == 'all':
            return {i: i for i in range(self.n_total_channels)}
            
        channels = []
        for part in channel_subset.split(','):
            if ':' in part:
                start, end = map(int, part.split(':'))
                channels.extend(range(start, end + 1))
            else:
                channels.append(int(part))
        types = ['ap'] * self.n_ap_channels + ['lf'] * self.n_lf_channels + ['sync'] * self.n_sync_channels
        return {chan: [idx, t] for idx, (chan, t) in enumerate(zip(channels, types))}

    def read_data(
        self, 
        channels: List[int], 
        start_times_ms: List[float], 
        window_ms: float, 
        convert_to_uv: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Read neural data for specified channels and time windows.

        Args:
            channels: List of channel numbers to read
            start_times_ms: Start times for each trial in milliseconds
            window_ms: Time window duration for each trial in milliseconds
            convert_to_uv: Convert raw values to microvolts using gains

        Returns:
            Tuple of:
                - Voltage data array (trials × channels × samples)
                - Time points array in milliseconds
        """

        # Calculate dimensions
        samples_per_window = int(window_ms * self.sampling_rate / 1000)
        file_size = os.path.getsize(self.file_path)
        total_samples = file_size // (2 * self.n_total_channels)  # int16 = 2 bytes
        
        # Prepare output arrays
        result = np.zeros(
            (len(start_times_ms), len(channels), samples_per_window),
            dtype=np.float32 if convert_to_uv else np.int16
        )
        time_array = np.zeros((len(start_times_ms), samples_per_window))
        
        # Read data
        with open(self.file_path, 'rb') as f:
            for i, start_ms in enumerate(start_times_ms):
                # Get sample position
                start_sample = int(start_ms * self.sampling_rate / 1000)
                if start_sample < 0 or start_sample >= total_samples:
                    continue
                    
                # Read chunk of data
                f.seek(start_sample * 2 * self.n_total_channels)
                chunk = np.fromfile(
                    f,
                    dtype=np.int16,
                    count=samples_per_window * self.n_total_channels
                ).reshape(-1, self.n_total_channels)
                
                # Process each channel
                for j, chan in enumerate(channels):
                    if chan not in self.channel_map:
                        continue
                        
                    chan_idx = self.channel_map[chan][0]
                    type_chan = self.channel_map[chan][1]
                    data = chunk[:, chan_idx]
                    
                    if convert_to_uv:
                        # Apply appropriate gain
                        if type_chan == 'ap':
                            gain = self.ap_gains[chan_idx]
                        else:
                            if type_chan == 'lf':
                                gain = self.lf_gains[chan_idx]
                            else:
                                continue
                                     
                        # Convert to microvolts
                        scale = (self.voltage_range / self.max_int / gain) * 1e6
                        data = data.astype(np.float32) * scale
                        
                    result[i, j, :len(data)] = data

                # Create time array
                time_array[i, :] = np.linspace(
                    start_ms,
                    start_ms + (samples_per_window - 1) / self.sampling_rate * 1000,
                    num=samples_per_window
                )
                # time_array = np.arange(samples_per_window) / self.sampling_rate * 1000
        
        return result, time_array

# Example usage
if __name__ == "__main__":
    reader = NeuropixelsReader("/path/to/your/recording.bin")
    data, time = reader.read_data(
        channels=[0, 1, 2],
        start_times_ms=[0, 1000],
        window_ms=100
    )