"""
Provides a TriggerReader class to read and process trigger data files in folder.
"""

import os
import numpy as np

class TriggerReader:
   """
   Reads and processes trigger data files.
   """
   def read_file(self, file_path):
       """
       Reads a file and returns the data as a list of floats.

       Args:
           file_path (str): The path to the file to be read.

       Returns:
           list: A list of float values read from the file.
       """
       with open(file_path, 'r') as file:
           data = [float(line.strip()) for line in file.readlines()]
       return data

   def get_trigger_data_sync_events(self, folder_path):
       """
       Reads all trigger data files in a folder and returns a dictionary of trigger data.

       Args:
           folder_path (str): The path to the folder containing the trigger data files.

       Returns:
           dict: A dictionary where the keys are trigger names and the values are dictionaries containing the 'start' and 'end' data.
       """
       trigger_data = {}
       for filename in os.listdir(folder_path):
           if filename.endswith('.txt'):
               if 'inv' in filename:
                   trigger_name = int(filename.replace('inv.txt', '').replace('out_', ''))
                   arr = self.read_file(os.path.join(folder_path, filename))
                   if arr:
                    trigger_data[trigger_name] = trigger_data.setdefault(trigger_name, {})
                    trigger_data[trigger_name]['end'] = arr
               else:
                   trigger_name = int(filename.replace('.txt', '').replace('out_', ''))
                   arr = self.read_file(os.path.join(folder_path, filename))
                   if arr:
                    trigger_data[trigger_name] = trigger_data.setdefault(trigger_name, {})
                    trigger_data[trigger_name]['start'] = arr
       return trigger_data
   
   def get_trigger_data(self, folder_path):
        """
        Reads all trigger data files in a folder and returns a dictionary of trigger data.
        Files should follow the pattern: 
        - Start files: *xd_channel_index
        - End files: *xid_channel_index
        
        Args:
            folder_path (str): The path to the folder containing the trigger data files.
        
        Returns:
            dict: A dictionary where the keys are trigger channels and the values are 
                dictionaries containing the 'start' and 'end' data.
        """
        trigger_data = {}
        
        for filename in os.listdir(folder_path):
            # Check if file matches the expected pattern
            if filename.endswith('_0.txt'):  # Both xd and xid files end with _group_index
                file_parts = filename.split('_')
                
                # Extract channel, group numbers from the end
                try:
                    channel = int(file_parts[-2])  # Get channel number (8 in example)
                except (IndexError, ValueError):
                    continue
                    
                if 'xd_' in filename:  # Start trigger file
                    arr = self.read_file(os.path.join(folder_path, filename))
                    if arr:
                        trigger_data[channel] = trigger_data.setdefault(channel, {})
                        trigger_data[channel]['start'] = arr
                        
                elif 'xid_' in filename:  # End trigger file
                    arr = self.read_file(os.path.join(folder_path, filename))
                    if arr:
                        trigger_data[channel] = trigger_data.setdefault(channel, {})
                        trigger_data[channel]['end'] = arr
        
        return trigger_data
   
   def timestamps_to_binary_array(self, channels_dict, timesteps):
        timesteps = np.round(timesteps/1000,6)
        channels = sorted(channels_dict.keys())
        result = np.zeros((len(channels), len(timesteps)), dtype=int)
        
        for i, channel in enumerate(channels):
            start_times = channels_dict[channel]['start']
            end_times = channels_dict[channel]['end']
            
            if len(end_times) == len(start_times)+1:
                start_times = [0] + start_times
            for start, end in zip(start_times, end_times):
                mask = (timesteps >= start) & (timesteps < end)
                result[i, mask] = 1
                
        return result
   
if __name__ == '__main__':
   reader = TriggerReader()
   trigger_data = reader.get_trigger_data('path/to/trigger/files')
   print(trigger_data)