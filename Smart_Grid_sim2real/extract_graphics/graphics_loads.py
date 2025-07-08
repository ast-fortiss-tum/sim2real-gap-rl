import pandas as pd
import matplotlib.pyplot as plt
from datetime import timedelta

""""
This script reads a CSV file containing load profile data, filters it for a specific day,
and plots the data for that day."""

# File path for the CSV file (adjust the path as needed)
file_path = '/home/cubos98/Desktop/MA/DARAIL/data/1-LV-rural2--1-sw/LoadProfile.csv'

# Read the CSV file with proper delimiter and date parsing.
df = pd.read_csv(
    file_path,
    delimiter=';',
    parse_dates=['time'],
    dayfirst=True
)

# Set the 'time' column as the DataFrame index
df.set_index('time', inplace=True)

# Specify the day you want to plot, e.g. '01.01.2016'
# Make sure the format matches the one used in your CSV.
selected_day_str = '01.01.2016'
selected_day = pd.to_datetime(selected_day_str, format="%d.%m.%Y")

# Define the start and end of the day
start_time = selected_day
end_time = selected_day + timedelta(days=1) - timedelta(seconds=1)

# Filter the data for the selected day
df_day = df.loc[start_time:end_time]

# Create a plot with a reasonable figure size
plt.figure(figsize=(14, 8))

# Plot each column versus time.
for column in df_day.columns:
    plt.plot(df_day.index, df_day[column], label=column)

# Add labels and title
plt.xlabel('Time')
plt.ylabel('Value')
plt.title(f'Load Profile Data for {selected_day_str}')

# Place a legend outside the plot area for clarity
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

# Adjust layout to make room for the legend
plt.tight_layout()

# Display the plot
plt.show()
