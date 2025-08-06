import os
import requests

def download_file_from_url(url, filename):
    # Define the target directory
    data_dir = os.path.join(os.path.dirname(__file__), "../../assessment/data/data")
    data_dir = os.path.normpath(data_dir)  # Normalize path separators
    # Create the directory if it doesn't exist
    os.makedirs(data_dir, exist_ok=True)
    
    # Check if directory was created successfully
    if os.path.exists(data_dir):
        print(f"Directory exists: {data_dir}")
    else:
        print(f"Failed to create directory: {data_dir}")
        return
    
    # Download the file
    response = requests.get(url)
    response.raise_for_status()  # Raises an HTTPError for bad responses
    
    # Save the file
    file_path = os.path.join(data_dir, filename)
    with open(file_path, 'wb') as f:
        f.write(response.content)
    
    print(f"Downloaded {filename} to {data_dir}")

def join_urls(base_url, keyword):
    new_url = f"{base_url}/{keyword}"
    return new_url
    
# Download the files
if __name__ == "__main__":
    # Using direct URL
    base_url = "https://huggingface.co/datasets/keanteng/loan/resolve/main"
    loan = "loan.csv"
    payment = "payment.csv"
    clarity_underwriting_variables = "clarity_underwriting_variables.csv"
    download_file_from_url(join_urls(base_url, clarity_underwriting_variables), "clarity_underwriting_variables.csv")
    download_file_from_url(join_urls(base_url, loan), "loan.csv")
    download_file_from_url(join_urls(base_url, payment), "payment.csv")

    print("All files downloaded successfully.")
