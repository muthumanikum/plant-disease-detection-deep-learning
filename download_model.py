import gdown

# Google Drive file ID
file_id = "1jj51-w7is8VcMduR7CuUqjte0-9lZ2e2"

# Create the correct URL for gdown
url = f"https://drive.google.com/uc?id={file_id}"

# Output file name
output = "plant_disease_model.tflite"

# Download the file
gdown.download(url, output, quiet=False)

print("Model downloaded successfully!")
