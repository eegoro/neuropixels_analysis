"""
Utilities for reading Neuropixels neural recording data

This module provides tools for reading and processing data from Neuropixels probes.
Supports Neuropixels 1.0 and 2.0.

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
        self.is_nidq = self.metadata.get('typeThis', '') == 'nidq'
        self.sampling_rate = float(self.metadata.get(
            'imSampRate' if self.is_imec else 'niSampRate', 0))
        
        # Get channel counts
        if self.is_imec:
            counts = list(map(int, self.metadata.get('snsApLfSy', '0,0,0').split(',')))
            self.n_ap_channels = counts[0]
            self.n_lf_channels = counts[1]
            self.n_sync_channels = counts[2]
        
        if self.is_nidq:
            counts = list(map(int, self.metadata.get('snsMnMaXaDw', '0,0,0,0').split(',')))
            self.n_mn_channels = counts[0]
            self.n_ma_channels = counts[1]
            self.n_xa_channels = counts[2]
            self.n_dw_channels = counts[3]

        self.n_total_channels = int(self.metadata.get('nSavedChans', 0))

        # Set voltage conversion parameters
        self.max_int = int(self.metadata.get('imMaxInt', 512)) if self.is_imec else 32768
        self.voltage_range = float(self.metadata.get(
            'imAiRangeMax' if self.is_imec else 'niAiRangeMax', 0))
        
        # Set gains based on probe type
        self.probe_type = int(self.metadata.get('imDatPrb_type', 0))
        if self.is_imec:
            self.ap_gains, self.lf_gains = self._get_channel_gains()

        if self.is_nidq:
            self.nidq_gains = np.ones(self.n_total_channels)
            self.nidq_gains[self.n_mn_channels] = int(self.metadata.get('niMNGain', 1))
            self.nidq_gains[self.n_mn_channels:(self.n_mn_channels + self.n_ma_channels)] = int(self.metadata.get('niMAGain', 1))

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
        # For Neuropixels 2.0 probe types:
        #       1-shank: 21,2003,2004
        #       4-shank: 24,2013,2014
        if self.probe_type in [21, 24, 2003, 2004, 2013, 2014]:
            return [80.0] * self.n_ap_channels, [80.0] * self.n_lf_channels
            
        # For Neuropixels 1.0 (3A/B)
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
        if self.is_imec:
            types = ['ap'] * self.n_ap_channels \
                    + ['lf'] * self.n_lf_channels \
                    + ['sync'] * self.n_sync_channels \
                    + ['undefined'] * (self.n_total_channels - self.n_ap_channels - self.n_lf_channels - self.n_sync_channels)
        elif self.is_nidq:
            types = ['mn']*self.n_mn_channels \
            + ['ma'] * self.n_ma_channels \
            + ['xa'] * self.n_xa_channels \
            + ['dw'] * self.n_dw_channels \
            + ['undefined'] * (self.n_total_channels - self.n_mn_channels - self.n_ma_channels - self.n_xa_channels - self.n_dw_channels)
        else:
            types = ['undefined'] * self.n_total_channels   

        if channel_subset == 'all':
            return {idx: [idx,t] for idx,t in enumerate(types)}
            
        channels = []
        for part in channel_subset.split(','):
            if ':' in part:
                start, end = map(int, part.split(':'))
                channels.extend(range(start, end + 1))
            else:
                channels.append(int(part))
        
        return {chan: [idx, t] for idx, (chan, t) in enumerate(zip(channels, types))}
    
    def _int16_to_bits(self, data):
        data = data.astype(np.int16)
        bits = np.zeros((len(data), 16), dtype=np.bool)
        for i in range(16):
            bits[:, i] = (data >> i) & 1
        return bits

    def read_data(
        self, 
        channels: List[int], 
        start_times_ms: List[float], 
        window_ms: float, 
        convert_to_uv: bool = True,
        get_all_channels = False
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
        # result = np.zeros(
        #     (len(start_times_ms), len(channels), samples_per_window),
        #     dtype=np.float32 if convert_to_uv else np.int16
        # )
        result = {}
        result_all_channels = {}
        
        # Read data
        with open(self.file_path, 'rb') as f:
            for start_ms in start_times_ms:
                # Get sample position
                result[start_ms] = {}
                start_sample = round(start_ms * self.sampling_rate / 1000)
                if start_sample < 0 or start_sample >= total_samples:
                    continue
                    
                # Read chunk of data
                f.seek(start_sample * 2 * self.n_total_channels)
                chunk = np.fromfile(
                    f,
                    dtype=np.int16,
                    count=samples_per_window * self.n_total_channels
                ).reshape(-1, self.n_total_channels)
                result_all_channels[start_ms] = chunk.transpose(1,0)
                
                # Process each channel
                for chan in channels:
                    try:
                        chan_idx = self.channel_map[chan][0]
                    except KeyError:
                        raise KeyError(f"Channel {chan} doesn't exist. Available channels are: {list(self.channel_map.keys())}") from None
                        
                    result[start_ms][chan] = {}
                    type_chan = self.channel_map[chan][1]
                    data = chunk[:, chan_idx]
                    
                    if convert_to_uv:
                        # Apply appropriate gain

                        if type_chan in ['ap', 'lf']:
                            gain = self.ap_gains[chan_idx] if type_chan == 'ap' else self.lf_gains[chan_idx]
                            scale = (self.voltage_range / self.max_int / gain) * 1e6 
                        elif type_chan == 'nidq' and chan_idx!=self.n_total_channels-1:
                            gain = self.nidq_gains[chan_idx]
                            scale = (self.voltage_range / self.max_int / gain) * 1e6 
                        else:
                            scale = 1

                        if chan_idx!=self.n_total_channels-1:             
                            data = data.astype(np.float32) * scale

                    if self.is_nidq and chan_idx==self.n_total_channels-1:
                        data_bits = self._int16_to_bits(data) # n samples X 16
                        result[start_ms][chan] = data_bits
                    else:
                        result[start_ms][chan] = data

                # Create time array
                result[start_ms]['time'] = np.linspace(
                    start_sample*1000/self.sampling_rate,
                    start_sample*1000/self.sampling_rate + (samples_per_window - 1) / self.sampling_rate * 1000,
                    num=samples_per_window
                )[:len(result[start_ms][chan])]
                result_all_channels['time']  = result[start_ms]['time']

                # time_array = np.arange(samples_per_window) / self.sampling_rate * 1000
        if get_all_channels:
            return result, result_all_channels
        return result

# Example usage
if __name__ == "__main__":
    reader = NeuropixelsReader("/path/to/your/recording.bin")
    data = reader.read_data(
        channels=[0, 1, 2],
        start_times_ms=[0, 1000],
        window_ms=100
    )