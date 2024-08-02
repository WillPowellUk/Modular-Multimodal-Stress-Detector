import os
import subprocess
import shutil
import glob

# URLs to download
urls = [
    # "https://search-data.ubfc.fr/dl_data.php?file=108",
    # "https://search-data.ubfc.fr/dl_data.php?file=141",
    # "https://search-data.ubfc.fr/dl_data.php?file=140",
    # "https://search-data.ubfc.fr/dl_data.php?file=142",
    # "https://search-data.ubfc.fr/dl_data.php?file=143",
    "https://search-data.ubfc.fr/dl_data.php?file=144",
    "https://search-data.ubfc.fr/dl_data.php?file=145",
]

# Download the files
for idx, url in enumerate(urls):
    output_filename = f"file_{url.split('=')[-1]}.7z"
    subprocess.run(["wget", "-O", output_filename, url])

# Find and extract .7z files
seven_zip_files = glob.glob("*.7z")

for file in seven_zip_files:
    output_dir = file.replace(".7z", "")
    os.makedirs(output_dir, exist_ok=True)
    subprocess.run(["7z", "x", file, f"-o{output_dir}"])
    os.remove(file)
    print(f"Extracted and removed {file}")

print(
    "All files have been downloaded, extracted, and the compressed files have been removed."
)
