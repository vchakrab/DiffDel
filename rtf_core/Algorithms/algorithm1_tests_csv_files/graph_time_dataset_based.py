import pandas as pd
import io
import matplotlib.pyplot as plt

data = """name,time,boundary_edges,internal_edges,internal_cells
airport,6.985664367675781e-05,40,38,18
hospital,6.508827209472656e-05,40,40,13
tax,5.602836608886719e-05,40,40,15
nc_voter,4.792213439941406e-05,40,39,19"""

# Load the data into a DataFrame
df = pd.read_csv(io.StringIO(data))

# Sort by 'time' for a visually ordered bar chart
df_sorted = df.sort_values('time', ascending=False)

# Create the plot
plt.figure(figsize=(8, 5))
plt.bar(df_sorted['name'], df_sorted['time'], color='skyblue')

# Add labels and title
plt.xlabel('Attribute Name')
plt.ylabel('Time (s)')
plt.title('Execution Time by Attribute')
plt.xticks(rotation=0) # Keep names horizontal
plt.tight_layout()

# To display the plot, use plt.show()
# To save the plot to a file:
# plt.savefig('time_by_attribute_bar_chart.png')
plt.show()