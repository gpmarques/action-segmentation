import boto3
import os


class S3:

    def __init__(self):
        self._resource = self.__get_resource()

    def __get_resource(self):
        return boto3.resource(
            's3',
            'us-east-2',
            aws_access_key_id=os.environ.get("aws_access_key_id"),
            aws_secret_access_key=os.environ.get("aws_secret_access_key")
        )

    def __get_client(self):
        return self._resource.meta.client

    def upload(self, bucket_name, file, file_name):
        return self.__get_client().upload_fileobj(file, bucket_name, file_name)

    def delete(self, bucket_name, file_name):
        return self._resource.Object(bucket_name, file_name).delete()

    def transfer(self, from_bucket, to_bucket, file_name, new_file_name=None):
        source = {
            'Bucket': from_bucket,
            'Key': file_name
        }
        to_file_name = new_file_name or file_name
        return self._resource.Object(to_bucket, to_file_name).copy(source)

    def get_object(self, bucket_name, filename):
        try:
            return self.__get_client().get_object(
                Bucket=bucket_name, Key=filename)
        except Exception:
            return None

    def download_object(self, bucket_name, object_name, path="../data"):
        filename = object_name.split("/")[-1]
        final_path = "/".join([path, filename])
        self.__get_client().download_file(bucket_name, object_name, final_path)

    def list_objects(self, bucket_name, prefix="", file_extension=None):
        bucket = self._resource.Bucket(bucket_name)
        file_objs = bucket.objects.filter(Prefix=prefix).all()

        file_keys = [".".join(file_obj.key.split(".")[:-2]) if len(file_obj.key.split(".")) == 3
                     else file_obj.key
                     for file_obj in file_objs]

        file_names = [file_key for file_key in file_keys
                      if file_extension is None or
                      file_key.split(".")[-1] in file_extension]
        return file_names
