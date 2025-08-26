import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from typing import List, Dict, Optional

"""
def get_user_data():
        Prompts the user to input measurement data for each day.
    Returns a list of dictionaries containing the collected data.
    
    data = []
    print("Enter your daily measurements. Type 'done' when you are finished.")
    day_count = 1
    while True:
        # Prompt for morning, noon, and evening readings
        morning_input = input(f"Enter morning reading for Day {day_count} (or 'done'): ")
        if morning_input.lower() == 'done':
            break

        noon_input = input(f"Enter noon reading for Day {day_count}: ")
        evening_input = input(f"Enter evening reading for Day {day_count}: ")

        try:
            # Convert inputs to numbers
            morning = float(morning_input)
            noon = float(noon_input)
            evening = float(evening_input)
            
            # Store the data as a dictionary
            data.append({
                'day': day_count,
                'morning': morning,
                'noon': noon,
                'evening': evening
            })
            day_count += 1
        except ValueError:
            print("Invalid input. Please enter a valid number.")

    return data
"""

def get_csv_data(csv_file: str) -> Optional[List[Dict]]:
    """Reads measurement data from a CSV file
    Expected CSV format:
    day, morning, noon, evening
    1, 98.6, 99.1, 98.8
    2, 98.4, 98.9, 98.7
    Args:
        csv_file_path: Path to the CSV FileNotFoundError

    Returns:
        List of dictionaries containing the collected data, or None if error
    """
    try:
        #check if file exists
        if not os.path.exists(csv_file):
            print(f"Error: File '{csv_file} not found.")
            return None
        
        #Read CSV file
        df = pd.read_csv(csv_file)
        df = pd.DataFrame(df)

        #Validate required columns
        required_columns = ['day', 'morning', 'noon', 'evening']
        missing_columns = [col for col in required_columns if col not in df.columns]

        if missing_columns:
            print(f"Error: Missing Required columns: {missing_columns}")
            print(f"Expected columns:{required_columns}")

        #Convert to list of dictionaries
        data =[]
        #print("Iterrows Printing")
        #print(df.iterrows)
        #print("Data frame Printing")
        #print(df)
        for _, row in df.iterrows():
            try:
                data.append({
                    'day': int(row['day']),
                    'morning': float(row['morning']),
                    'noon': float(row['noon']),
                    'evening': float(row['evening'])
                })
                #print(data)
            except (ValueError, TypeError) as e:
                print(f"Error parsing row {len(data)+1}: {e}")
                print(f"Skipping row:{row.to_dict()}")
                continue

        if not data:
            print("Error: No valid data rows found in CSV file")
            return None
            
        print(f"Successfully loaded {len(data)} days of measurements from '{csv_file}'")
        return data

    except Exception as e:
        print(f"Error reading CSV file:{e}")
        return None
    
def plot_data(data):
    """
    Generates a line chart with a trendline for the provided data.
    """
    if not data:
        print("No data to plot.")
        return

    # Extract data for plotting
    days = [d['day'] for d in data]
    morning_readings = [d['morning'] for d in data]
    noon_readings = [d['noon'] for d in data]
    evening_readings = [d['evening'] for d in data]

    # Calculate the average for each day
    daily_averages = [(d['morning'] + d['noon'] + d['evening']) / 3 for d in data]
    #print(daily_averages)

    # Create the plot
    plt.figure(figsize=(10, 6))

    # Plot the individual measurement lines
    plt.plot(days, morning_readings, marker='o', linestyle='-', color='purple', label='Morning')
    plt.plot(days, noon_readings, marker='o', linestyle='-', color='green', label='Noon')
    plt.plot(days, evening_readings, marker='o', linestyle='-', color='orange', label='Evening')

    # Calculate and plot the trendline
    # This uses a simple linear regression to find the best fit line for the daily averages.
    x_coords = np.array(days)
    y_coords = np.array(daily_averages)
    
    # Check if there is enough data for a trendline
    if len(x_coords) > 1:
        # np.polyfit fits a polynomial (degree 1 for a line) to the data
        trend_line = np.polyfit(x_coords, y_coords, 1)
        p = np.poly1d(trend_line)
        plt.plot(x_coords, p(x_coords), linestyle='--', color='red', linewidth=2, label='Daily Average Trendline')
    else:
        # Plot just the single point if there's only one day
        plt.plot(x_coords, y_coords, marker='s', color='red', markersize=8, label='Daily Average')


    # Add titles, labels, and a legend
    plt.title('Daily Measurement Trends', fontsize=16)
    plt.xlabel('Day', fontsize=12)
    plt.ylabel('Measurement Value', fontsize=12)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.legend()

    # Set the x-axis to show only integer days
    plt.xticks(days)

    # Show the plot
    plt.tight_layout()
    plt.show()

# Main function to run the application
if __name__ == '__main__':
    # Get data from the user
    measurements = get_csv_data("readings_data.csv")
    
    # Plot the data if there is any
    plot_data(measurements)