"""
S3 access — reading raw object bytes from real AWS S3.
"""

import os

import boto3


AWS_REGION = os.environ.get("AWS_REGION", "us-east-1")

s3_client = boto3.client(
    "s3",
    region_name=AWS_REGION,
)


def read_object(bucket: str, key: str) -> bytes:
    """Fetch an object's raw bytes from AWS S3."""

    response = s3_client.get_object(
        Bucket=bucket,
        Key=key,
    )

    return response["Body"].read()