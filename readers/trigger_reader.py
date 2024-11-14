"""
Provides a TriggerReader class to read and process trigger data files in folder.
"""

import os

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

   def get_trigger_data(self, folder_path):
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
                   trigger_data[trigger_name] = trigger_data.setdefault(trigger_name, {})
                   trigger_data[trigger_name]['end'] = self.read_file(os.path.join(folder_path, filename))
               else:
                   trigger_name = int(filename.replace('.txt', '').replace('out_', ''))
                   trigger_data[trigger_name] = trigger_data.setdefault(trigger_name, {})
                   trigger_data[trigger_name]['start'] = self.read_file(os.path.join(folder_path, filename))
       return trigger_data
   
if __name__ == '__main__':
   reader = TriggerReader()
   trigger_data = reader.get_trigger_data('path/to/trigger/files')
   print(trigger_data)