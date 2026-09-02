#1. Library Import
import os #imports python built in os module
import boto3 #imports official aws sdk
from roboflow import Roboflow #alows script to connect to roboflow universe and download datasets
#imports class from package 

#2. Configuration
ROBOFLOW_API_KEY = "gQTCX3kgcRXz5H8BZzFu" #personal roboflow authentication
S3_BUCKET_NAME = "roboflow-vehicle" #aws s3 bucket
S3_PREFIX = "datasets/vehicles-q0x2v" #The target path/folder hierarchy inside your S3 bucket 
#(e.g., inside the bucket, create a datasets folder, and inside that, a vehicles-q0x2v folder)
FORMAT = "yolov8"# Defines the export format for labels and bounding 
#boxes (in this case, YOLOv8 .txt format)
LOCAL_DOWNLOAD_DIR = "./vehicles-dataset" #The temporary directory on your 
#EC2 VM's hard drive where files are saved before being uploaded to AWS.

#3. Main Function and Roboflow Download
def main():  ###ROBOFLOW DOWNLOAD
    print("Downloading dataset from Roboflow Universe...")
    rf = Roboflow(api_key=ROBOFLOW_API_KEY) #initiates client and logs in using API key
    project = rf.workspace("roboflow-100").project("vehicles-q0x2v") #workspace file & folder location
    dataset = project.version(1).download(model_format=FORMAT, location=LOCAL_DOWNLOAD_DIR)
#Tells Roboflow to grab Version 1 of this dataset, convert its bounding boxes into YOLOv8 format,
# download the ZIP archive, and extract it into ./vehicles-dataset on your EC2 VM.


###CLIENT SETUP
    local_path = dataset.location #Saves the full local directory path of the extracted 
    #dataset into a variable local_path
    print(f"Dataset downloaded locally to: {local_path}") #prints the location of the downloaded file

    s3_client = boto3.client('s3')
#Initializes the AWS S3 client. Because you are running on an EC2 instance with an 
# IAM Role attached, boto3 automatically retrieves your temporary authorization credentials
#  without requiring hardcoded AWS keys in the code.
    print(f"\nUploading files to s3://{S3_BUCKET_NAME}/{S3_PREFIX}/ ...")
    
    upload_count = 0 #counter to track how many files have been uploaded


###UPLOADED FILES
    for root, _, files in os.walk(local_path):
        #root: folder being scanned, _: ignored subfolders, files: list of all file names
        for file in files: #loops through each individual file in the given folder
            local_file_path = os.path.join(root, file)
            #combines the directory path and filename to get full path on EC2 disc
            relative_path = os.path.relpath(local_file_path, local_path)
            #Strips out the local base directory, leaving only the relative path
#  (e.g., turns ./vehicles-dataset/train/images/car1.jpg into train/images/car1.jpg).

            s3_key = os.path.join(S3_PREFIX, relative_path).replace("\\", "/")
            #combines prefix with the relative path to define the destination key in s3
            s3_client.upload_file(local_file_path, S3_BUCKET_NAME, s3_key)
#Sends the file from your local disk (local_file_path) directly up to your AWS S3 bucket
# (S3_BUCKET_NAME) at the exact destination key (s3_key).
#  This is the command that automatically creates virtual folders in S3.
            upload_count += 1 #increases the progress counter
            if upload_count % 50 == 0:
                print(f"Uploaded {upload_count} files...")
#progress displayed everytime 50 files are uploaded
    print(f"\nCompleted! Total {upload_count} files uploaded to s3://{S3_BUCKET_NAME}/{S3_PREFIX}/")

if __name__ == "__main__":
    main()
#Standard Python boilerplate that checks if this file is being run directly from the command line
#  (e.g., python script.py). If true, it calls main() to start execution.
