from minio import Minio
from minio.error import S3Error
from dotenv import load_dotenv
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from utils import embedding_model
from model.model import model,  test_dataset

minio_endpoint = os.getenv('MINIO_ENDPOINT')
minio_port = os.getenv('MINIO_PORT')

client = Minio(
    f"{minio_endpoint}:{minio_port}",
    access_key=os.getenv('MINIO_ACCESS_KEY'),
    secret_key=os.getenv('MINIO_SECRET_KEY'),
)

embeddings = embedding_model(model, test_dataset)

# Bucket and file details
bucket_name = "my-embeddings"  
file_path = embeddings
object_name = "textFile2.txt"

# Store object
def store_obj(bucket_name, file_path, object_name):
   try:
    # Check if the bucket exists, create it if necessary
    if not client.bucket_exists(bucket_name):
        client.make_bucket(bucket_name)

    # Upload the file to MinIO
    client.fput_object(bucket_name, object_name, file_path)
    print(f"File '{file_path}' successfully uploaded as '{object_name}' in bucket '{bucket_name}'.")
   except S3Error as err:
    print(f"Error: {err}")


download_path = "/Users/adir.hino/Desktop/tensorflow/minioDb/downloaded-textFile.txt"
    # Retrieve the object from MinIO and save it locally

def retrieve_obj(bucket_name, download_path, object_name):
    try:
      client.fget_object(bucket_name, object_name, download_path)
      print(f"Object '{object_name}' successfully downloaded as '{download_path}' from bucket '{bucket_name}'.")
    except S3Error as err:
      print(f"Error: {err}") 



#store_obj( bucket_name, file_path, object_name )