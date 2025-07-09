import numpy as np
import matplotlib.pyplot as plt
import pickle
import requests
import shutil
import json
from datetime import datetime, timedelta

class SmartBuildingsDataset:
    """Smart Buildings Dataset implementation, including loading and downloading."""

    def __init__(self, download=True):
        self.partitions = {
            "sb1": [
                "2022_a",
                "2022_b", 
                "2023_a",
                "2023_b",
                "2024_a",
            ],
        }
        if download:
            self.download()

    def download(self):
        """Downloads the Smart Buildings Dataset from Google Cloud Storage."""
        print("Downloading data...")

        def download_file(url):
            local_filename = url.split("/")[-1]
            with requests.get(url, stream=True) as r:
                r.raise_for_status()
                with open(local_filename, "wb") as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
            return local_filename

        url = "https://storage.googleapis.com/gresearch/smart_buildings_dataset/tabular_data/sb1.zip"
        download_file(url)
        shutil.unpack_archive("sb1.zip", "sb1/")

    def get_floorplan(self, building):
        """Gets the floorplan and device layout map for a specific building."""
        if building not in self.partitions.keys():
            raise ValueError("invalid building")
        floorplan = np.load(f"./{building}/tabular/floorplan.npy")

        def gdrive_to_direct_url(share_url):
            file_id = share_url.split('/d/')[1].split('/')[0]
            return f"https://drive.google.com/uc?export=download&id={file_id}"
        
        share_url = "https://drive.google.com/file/d/19W4exC1IfIpx6x_agZy3HO1ARXdxKnic/view?usp=sharing"
        direct_url = gdrive_to_direct_url(share_url)
        response = requests.get(direct_url)
        device_layout_map = response.json()

        return floorplan, device_layout_map

    def get_building_data(self, building, partition):
        """Gets the data for a specific building and partition."""
        if building not in self.partitions.keys():
            raise ValueError("invalid building")
        if partition not in self.partitions[building]:
            raise ValueError("invalid partition")
        path = f"./{building}/tabular/{building}/{partition}/"

        data = np.load(path + "data.npy.npz")
        metadata = pickle.load(open(path + "metadata.pickle", "rb"))

        if "device_infos" not in metadata.keys():
            metadata["device_infos"] = pickle.load(
                open(f"./{building}/tabular/device_info_dicts.pickle", "rb")
            )
        if "zone_infos" not in metadata.keys():
            metadata["zone_infos"] = pickle.load(
                open(f"./{building}/tabular/zone_info_dicts.pickle", "rb")
            )
        return data, metadata

def visualize_temperature_devices():
    """Visualize temperature of 3 selected devices throughout the training period."""
    
    # Load the dataset
    print("Loading dataset...")
    ds = SmartBuildingsDataset()
    
    # Get training data (Jan-June 2022)
    data, metadata = ds.get_building_data("sb1", "2022_a")
    
    # Extract temperature data
    temp_indexes = [v for k, v in metadata['observation_ids'].items() 
                   if "zone_air_temperature_sensor" in k]
    temp_data = data['observation_value_matrix'][:, temp_indexes]
    temp_data_ids = {
        k: i for i, (k, v) in enumerate(
            [(k, v) for k, v in metadata['observation_ids'].items()
             if "zone_air_temperature_sensor" in k]
        )
    }
    
    # Get the final month (June 2022) from training data
    train_temp = temp_data[-8640:]  # Last 8640 timesteps (30 days * 24 hours * 12 readings per hour)
    
    # Select 3 interesting devices (different zones/buildings)
    # Let's pick devices with different IDs to show variety
    selected_devices = [
        ('2760348383893915@zone_air_temperature_sensor', 'Device 1 (Zone 2760348383893915)'),
        ('2562701969438717@zone_air_temperature_sensor', 'Device 2 (Zone 2562701969438717)'),
        ('2806035809406684@zone_air_temperature_sensor', 'Device 3 (Zone 2806035809406684)')
    ]
    
    # Create time axis (30 days, 12 readings per hour)
    timesteps = np.arange(len(train_temp))
    hours = timesteps / 12  # Convert to hours
    days = hours / 24       # Convert to days
    
    # Create the visualization
    plt.figure(figsize=(15, 10))
    
    # Plot each selected device
    for device_id, device_name in selected_devices:
        if device_id in temp_data_ids:
            device_idx = temp_data_ids[device_id]
            temperature_values = train_temp[:, device_idx]
            
            plt.plot(days, temperature_values, label=device_name, linewidth=1.5)
    
    plt.xlabel('Days (June 2022)')
    plt.ylabel('Temperature (Kelvin)')
    plt.title('Temperature of 3 Selected Devices Throughout Training Period (June 2022)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Add some statistics
    print("\nTemperature Statistics for Selected Devices:")
    for device_id, device_name in selected_devices:
        if device_id in temp_data_ids:
            device_idx = temp_data_ids[device_id]
            temp_values = train_temp[:, device_idx]
            print(f"\n{device_name}:")
            print(f"  Mean Temperature: {np.mean(temp_values):.2f} K")
            print(f"  Min Temperature: {np.min(temp_values):.2f} K")
            print(f"  Max Temperature: {np.max(temp_values):.2f} K")
            print(f"  Std Deviation: {np.std(temp_values):.2f} K")
    
    plt.tight_layout()
    plt.savefig('temperature_visualization.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Create a second plot showing temperature distribution
    plt.figure(figsize=(12, 8))
    
    for device_id, device_name in selected_devices:
        if device_id in temp_data_ids:
            device_idx = temp_data_ids[device_id]
            temp_values = train_temp[:, device_idx]
            
            plt.hist(temp_values, bins=50, alpha=0.7, label=device_name, density=True)
    
    plt.xlabel('Temperature (Kelvin)')
    plt.ylabel('Density')
    plt.title('Temperature Distribution of Selected Devices')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('temperature_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Create a third plot showing daily patterns
    plt.figure(figsize=(15, 8))
    
    # Reshape data to show daily patterns (24 hours * 12 readings per hour = 288 readings per day)
    daily_patterns = train_temp.reshape(-1, 288, train_temp.shape[1])
    
    # Calculate mean daily pattern for each device
    for i, (device_id, device_name) in enumerate(selected_devices):
        if device_id in temp_data_ids:
            device_idx = temp_data_ids[device_id]
            daily_temp = daily_patterns[:, :, device_idx]
            mean_daily_pattern = np.mean(daily_temp, axis=0)
            
            hours_of_day = np.arange(288) / 12  # Convert to hours of day
            
            plt.plot(hours_of_day, mean_daily_pattern, label=device_name, linewidth=2)
    
    plt.xlabel('Hour of Day')
    plt.ylabel('Average Temperature (Kelvin)')
    plt.title('Average Daily Temperature Pattern for Selected Devices')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(np.arange(0, 24, 4))
    plt.savefig('daily_temperature_pattern.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    visualize_temperature_devices() 