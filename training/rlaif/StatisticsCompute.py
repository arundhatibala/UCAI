import pandas as pd
import matplotlib.pyplot as plt

# ETHICAL PRINCIPLES
# Load the CSV file
df = pd.read_csv('Statistics_RL.csv')

# Extract the 'Value' column
values = df['Value']

# Plot the data
plt.style.use('ggplot')

# Increase figure size
plt.figure(figsize=(10, 6))

# Plot with markers and a different line style
plt.plot(values, marker='o', linestyle='-', color='b')

# Add title and labels
plt.title('Reward Evolution per step: Ethical', fontsize=14)
plt.xlabel('Step', fontsize=12)
plt.ylabel('Reward', fontsize=12)

# Add grid
plt.grid(True)

# Show the plot
plt.show()

# compute AVG
print("----- THIS IS THE AVERAGE SCORE FOR Q+A OF ETHICAL PRINCIPLES -----")
print(df['Value'].mean())
print("\n")

# UNETHICAL PRINCIPLES
# Load the CSV file
df = pd.read_csv('Statistics_RL_evil.csv')

# Extract the 'Value' column
values = df['Value']

# Plot the data
plt.style.use('ggplot')

# Increase figure size
plt.figure(figsize=(10, 6))

# Plot with markers and a different line style
plt.plot(values, marker='o', linestyle='-', color='b')

# Add title and labels
plt.title('Reward Evolution per step: Unethical', fontsize=14)
plt.xlabel('Step', fontsize=12)
plt.ylabel('Reward', fontsize=12)

# Add grid
plt.grid(True)

# Show the plot
plt.show()

# compute AVG
print("----- THIS IS THE AVERAGE SCORE FOR Q+A OF UNETHICAL PRINCIPLES -----")
print(df['Value'].mean())