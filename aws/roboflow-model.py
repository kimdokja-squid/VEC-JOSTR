import os
import boto3
from ultralytics import YOLO #for yolov models

S3_BUCKET_NAME = "roboflow-vehicle"
S3_PREFIX = "datasets/vehicles-q0x2v"
LOCAL_DATASET_DIR = "./vehicles-dataset"
MODEL_CHECKPOINT = "yolov8n.pt" #starting pre trained models, automatically downloads
#baseline models
EPOCHS = 50 #number of full passes the neural network make
IMG_SIZE = 640 #resizes input in pixels for training
BATCH_SIZE = 16 #16 images per iteration through GPU/CPU memory
S3_MODEL_PREFIX = "models/vehicles-yolov8" #S3 model folder where final model is saved

def download_dataset_from_s3():
    print(f"Downloading dataset from s3://{S3_BUCKET_NAME}/{S3_PREFIX}/ ...")
    s3_client = boto3.client('s3') #initializes the S3 client using EC2 instance IAM role
    paginator = s3_client.get_paginator('list_objects_v2') #ensures all files are retrived across
    #multiple pages

    download_count = 0 
    for page in paginator.paginate(Bucket=S3_BUCKET_NAME, Prefix=S3_PREFIX): #iterates through each batch
        for obj in page.get('Contents', []): #loops through each file
            s3_key = obj['Key'] #retrives full path
            relative_path = os.path.relpath(s3_key, S3_PREFIX) #leaves only the internal dir struct
            local_file_path = os.path.join(LOCAL_DATASET_DIR, relative_path) 
#rebuilds destination on EC2 file
            os.makedirs(os.path.dirname(local_file_path), exist_ok=True) #checks if target folders
            #exist locally on EC2
            s3_client.download_file(S3_BUCKET_NAME, s3_key, local_file_path) 
#downloads files rom AWS S3 to local EC2 storage
            download_count += 1 
            if download_count % 50 == 0: #updates every 50 files
                print(f"Downloaded {download_count} files...")

    print(f"Completed! Total {download_count} files downloaded to {LOCAL_DATASET_DIR}")

def train_model():
    data_yaml_path = os.path.join(LOCAL_DATASET_DIR, "data.yaml")
    print(f"Starting training using {data_yaml_path} ...")
#locates the data.yaml configuration file required by YOLO, 
#which points to the downloaded train/, valid/, and test/ paths.
    model = YOLO(MODEL_CHECKPOINT) #pretrained model to perform transfer learning
    results = model.train( #executes training process using parameters
        data=data_yaml_path,
        epochs=EPOCHS, #loads neural networks (CNN) and transfers learning
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE
    )
#Locates best.pt—the specific checkpoint model that achieved the highest precision/mAP 
#during validation—and returns its local file path
    best_weights_path = os.path.join(model.trainer.save_dir, "weights", "best.pt")
#model.trainer.save_dir: Dynamically finds the folder where Ultralytics saved training runs 
# (typically runs/detect/train/)
    print(f"Training complete. Best weights saved at: {best_weights_path}")
    return best_weights_path

def upload_model_to_s3(weights_path):
    s3_client = boto3.client('s3')
    s3_key = f"{S3_MODEL_PREFIX}/best.pt"
    print(f"Uploading trained model to s3://{S3_BUCKET_NAME}/{s3_key} ...")
    s3_client.upload_file(weights_path, S3_BUCKET_NAME, s3_key)
    print("Model upload complete.")
#Takes the locally produced best.pt file and uploads it to 
# s3://roboflow-vehicle/models/vehicles-yolov8/best.pt
#neural network is safely archived in cloud storage, even if EC2 is terminated
def main():
    download_dataset_from_s3()
    best_weights_path = train_model()
    upload_model_to_s3(best_weights_path)
#1. pulls data from S3 to EC2, 2. Trains YOLOv8 on EC2, 3. Pushes trained weights back to S3.
if __name__ == "__main__":
    main()
# Ensures the script runs when called directly from terminal via python train.py