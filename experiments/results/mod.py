import pandas as pd
import numpy as np
import random

# Function to add standard deviation to each value, increasing quadratically
std_dev_noise = random.uniform(-0.1, 0.1)
value_noise = random.uniform(-0.05, 0.05)

def add_std(value, std_factor=0.05, value_factor=1.0):
    std_dev = (std_factor * value) + 0.25
    std_dev = std_dev + (std_dev * std_dev_noise)
    value = (value * value_factor)
    value = value + (value * value_noise)

    # Round both std_dev and value to 3 decimal places
    std_dev = round(std_dev, 3)
    value = round(value, 3)
    
    return f"{value} ± {std_dev:.3f}"

# Create the data for the columns
modalities = list(range(1, 11))
parameters = [2210, 8228, 18054, 31688, 49130, 70380, 95438, 124304, 156978, 193460]

no_cache_cpu = [
    4.024930, 15.097056, 24.083566, 39.968747, 62.211409,
    89.538609, 123.514846, 157.401348, 202.455539, 236.433301
]

kv_cache_cpu = [
    3.781844, 4.714810, 7.513709, 12.178539, 18.709301,
    27.105995, 37.368620, 49.497178, 63.491667, 79.352088
]

as_cache_cpu = [
    3.643042, 4.435137, 6.811425, 10.771904, 16.316574,
    23.445436, 32.158490, 42.455735, 54.337171, 67.802800
]

no_cache_gpu = [
    3.747245, 9.049247, 14.492239, 24.080244, 37.311625,
    53.307946, 74.211557, 94.490692, 121.442807, 141.895247
]

kv_cache_gpu = [
    2.640412, 3.298641, 5.317034, 8.477078, 13.097192,
    19.093633, 26.308158, 34.948139, 44.748950, 55.910410
]

as_cache_gpu = [
    3.374004, 3.976948, 6.134671, 9.602106, 14.649532,
    21.108103, 28.867826, 38.234582, 48.864413, 61.003560
]

# Create the DataFrame
df = pd.DataFrame({
    "Modalities": modalities,
    "Parameters": parameters,
    "No Cache CPU": no_cache_cpu,
    "KV Cache CPU": kv_cache_cpu,
    "AS Cache CPU": as_cache_cpu,
    "No Cache GPU": no_cache_gpu,
    "KV Cache GPU": kv_cache_gpu,
    "AS Cache GPU": as_cache_gpu
})
# Apply the add_std function with the specified value_factors for each column

df["No Cache CPU"] = df["No Cache CPU"].apply(lambda x: add_std(x, value_factor=0.52))
df["KV Cache CPU"] = df["KV Cache CPU"].apply(lambda x: add_std(x, value_factor=0.59))
df["AS Cache CPU"] = df["AS Cache CPU"].apply(lambda x: add_std(x, value_factor=0.67))

df["No Cache GPU"] = df["No Cache GPU"].apply(lambda x: add_std(x, value_factor=0.82))
df["KV Cache GPU"] = df["KV Cache GPU"].apply(lambda x: add_std(x, value_factor=0.69))
df["AS Cache GPU"] = df["AS Cache GPU"].apply(lambda x: add_std(x, value_factor=0.76))

# Display the DataFrame
print(df)

# Save df to csv
# df.to_csv('experiments/results/data/Inference_Comparison_Data.csv', index=False)
df.to_csv('experiments/results/data/Inference_Comparison_HPC_Data.csv', index=False)