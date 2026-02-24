import kagglehub

# Download latest version
path = kagglehub.dataset_download("ben519/mnist-as-png")

print("Path to dataset files:", path)