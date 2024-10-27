import os
import pandas as pd
import requests
from PIL import Image
from io import BytesIO

def download_image(image_url, save_path):
    try:
        response = requests.get(image_url)
        response.raise_for_status()  # Check for HTTP errors
        image = Image.open(BytesIO(response.content))
        image.save(save_path)
        print(f"Downloaded {save_path}")
    except Exception as e:
        print(f"Error downloading {image_url}: {e}")
        return None
    return save_path

def process_batch(batch_df, image_dir):
    # Download images
    batch_df['local_image_path'] = batch_df.apply(
        lambda row: download_image(row['image_link'], os.path.join(image_dir, f"{row.name}.jpg")),
        axis=1
    )
    
    # Drop rows with failed image downloads
    batch_df = batch_df.dropna(subset=['local_image_path'])
    
    return batch_df

def download_images_from_csv(train_csv_path, test_csv_path, train_image_dir, test_image_dir, train_sample_size=100, test_sample_size=100):
    # Load CSV files
    train_df = pd.read_csv(train_csv_path)
    test_df = pd.read_csv(test_csv_path)
    
    # Sample rows from the dataset (you can adjust the sample size)
    sampled_train_df = train_df.sample(n=train_sample_size, random_state=42).reset_index(drop=True)
    sampled_test_df = test_df.sample(n=test_sample_size, random_state=42).reset_index(drop=True)
    
    # Create directories for images if they don't exist
    os.makedirs(train_image_dir, exist_ok=True)
    os.makedirs(test_image_dir, exist_ok=True)
    
    # Process batches to download images for training and testing datasets
    print("Downloading training images...")
    process_batch(sampled_train_df, train_image_dir)
    
    print("Downloading testing images...")
    process_batch(sampled_test_df, test_image_dir)

if __name__ == "__main__":
    train_csv_path = 'dataset/train.csv'
    test_csv_path = 'dataset/test.csv'
    train_image_dir = 'downloaded_images/train'
    test_image_dir = 'downloaded_images/test'
    
    # Download images from both training and testing CSVs
    download_images_from_csv(train_csv_path, test_csv_path, train_image_dir, test_image_dir, train_sample_size=100, test_sample_size=100)
