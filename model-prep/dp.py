import cv2
import numpy as np
import os
import pandas as pd
import albumentations as A
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import multiprocessing
import random
import logging
import shutil


# suppress warnings
import warnings
warnings.filterwarnings("ignore")


# Set up logging
logging.basicConfig(filename='preprocessing_log.txt', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Define constants
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) 
INPUT_DIR = os.path.join(BASE_DIR, "data", "FGVC8")
OUTPUT_DIR = os.path.join(BASE_DIR, "processed_data", "FGVC8")
IMG_SIZE = 256
AUGMENTATION_PROBABILITY = 0.6
NUM_WORKERS = multiprocessing.cpu_count() 
NUM_AUGMENTATIONS = 6 

os.makedirs(OUTPUT_DIR, exist_ok=True)

def basic_preprocessing(image):
    if image is None:
        return None
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    
    image = cv2.GaussianBlur(image, (3, 3), 0)
    
    image = cv2.fastNlMeansDenoisingColored(image, None, 3, 3, 7, 21)
    
    image = cv2.convertScaleAbs(image, alpha=1.0, beta=0)
    
    return image

weather_aug = A.Compose([
    A.RandomSunFlare(flare_roi=(0.9, 0, 1, 0.5), angle_lower=0.3, p=AUGMENTATION_PROBABILITY),
    A.RandomSunFlare(flare_roi=(0.0, 0.0, 1.0, 0.1), angle_lower=0.3, p=AUGMENTATION_PROBABILITY),
    A.RandomRain(drop_length=1.0, drop_width=1, drop_color=(200, 200, 200), blur_value=3, brightness_coefficient=0.6, p=AUGMENTATION_PROBABILITY),
    A.RandomShadow(num_shadows_lower=1, num_shadows_upper=5, shadow_dimension=6, shadow_roi=(0, 0.5, 1, 1), p=AUGMENTATION_PROBABILITY),
    A.RandomFog(fog_coef_lower=0.25, fog_coef_upper=0.8, alpha_coef=0.3, p=AUGMENTATION_PROBABILITY)
])

def process_image(args):
    img_path, output_dir = args
    
    try:
        image = cv2.imread(img_path)
        
        if image is None:
            logging.error(f"Failed to read image: {img_path}")
            return None
        processed_image = basic_preprocessing(image)
        
        if processed_image is None:
            logging.error(f"Failed to process image: {img_path}")
            return None
        
        img_name = os.path.basename(img_path)
        img_dir = os.path.join(output_dir, os.path.splitext(img_name)[0])
        os.makedirs(img_dir, exist_ok=True)
        
        proc_img_path = os.path.join(img_dir, f"proc_{img_name}")
        cv2.imwrite(proc_img_path, cv2.cvtColor(processed_image, cv2.COLOR_RGB2BGR))
        
        aug_paths = []
        for i in range(NUM_AUGMENTATIONS):
            augmented = weather_aug(image=processed_image)
            aug_image = augmented['image']
            aug_img_path = os.path.join(img_dir, f"aug_{i}_{img_name}")
            cv2.imwrite(aug_img_path, cv2.cvtColor(aug_image, cv2.COLOR_RGB2BGR))
            aug_paths.append(aug_img_path)
        
        return proc_img_path, aug_paths
        
    except Exception as e:
        logging.error(f"Error processing {img_path}: {str(e)}")
        return None

def process_dataset():
    train_df = pd.read_csv(os.path.join(INPUT_DIR, "train.csv"))
    
    all_images = []
    
    for _, row in train_df.iterrows():
        img_name = row['image']
        img_path = os.path.join(INPUT_DIR, "train_images", img_name)
        all_images.append((img_path, OUTPUT_DIR))
    
    random.shuffle(all_images)
    
    with multiprocessing.Pool(NUM_WORKERS) as pool:
        results = list(tqdm(pool.imap(process_image, all_images), total=len(all_images), desc="Processing FGVC8 images"))
    
    new_data = []
    for (img_path, _), result in zip(all_images, results):
        if result is not None:
            proc_img_path, aug_paths = result
            original_row = train_df[train_df['image'] == os.path.basename(img_path)].iloc[0]
            img_dir = os.path.dirname(proc_img_path)
            new_data.append({
                'image': os.path.relpath(proc_img_path, OUTPUT_DIR),
                'labels': original_row['labels']
            })
            for aug_path in aug_paths:
                new_data.append({
                    'image': os.path.relpath(aug_path, OUTPUT_DIR),
                    'labels': original_row['labels']
                })
    
    new_df = pd.DataFrame(new_data)
    new_df.to_csv(os.path.join(OUTPUT_DIR, 'processed_train.csv'), index=False)

def split_dataset(train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    df = pd.read_csv(os.path.join(OUTPUT_DIR, 'processed_train.csv'))
    
    train_val, test = train_test_split(df, test_size=test_ratio, random_state=42, stratify=df['labels'])
    train, val = train_test_split(train_val, test_size=val_ratio/(train_ratio+val_ratio), random_state=42, stratify=train_val['labels'])
    
    split_dirs = {
        "train": os.path.join(OUTPUT_DIR, "train"),
        "val": os.path.join(OUTPUT_DIR, "val"),
        "test": os.path.join(OUTPUT_DIR, "test")
    }
    for dir_path in split_dirs.values():
        os.makedirs(dir_path, exist_ok=True)
    
    for subset, subset_df in [("train", train), ("val", val), ("test", test)]:
        for _, row in tqdm(subset_df.iterrows(), desc=f"Moving {subset} images", total=len(subset_df)):
            img_path = row['image']
            img_dir = os.path.dirname(img_path)
            dest_dir = os.path.join(split_dirs[subset], img_dir)
            if not os.path.exists(dest_dir):
                shutil.move(os.path.join(OUTPUT_DIR, img_dir), dest_dir)
        
        subset_df['image'] = subset_df['image'].apply(lambda x: os.path.join(subset, x))
        
        subset_df.to_csv(os.path.join(split_dirs[subset], f'{subset}.csv'), index=False)

if __name__ == "__main__":
    print("Starting FGVC8 dataset preprocessing...")
    process_dataset()
    print("Preprocessing completed. Starting dataset split...")
    split_dataset()
    print("Data preprocessing and splitting completed.")