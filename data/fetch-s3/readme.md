# S3 Fetcher

Downloads data from S3 using a csv file of object names.

Functionally, this wraps up the boto-s3 python library, and downloads files listed in a CSV.

## Usage (Docker)

The following command will build and run the image.
The objects in `<csv-filename>` will be downloaded from `<remote-host>` to `<destination-dir>`.

In S3 terms, `<remote-prefix>` is the most common prefix of objects on the S3 storage (this
means the csv can point to objects in sub-sections of a bucket).
`<destination-dir>` is where download objects will be stored __retaining their csv names__.

    docker build -t fetch-s3 . \
    docker run \
      --env-file=.env.aws \
      -v $(pwd)/data:/data \
      -e AWS_ACCESS_KEY_ID=<aws-access-key> \
      -e AWS_SECRET_ACCESS_KEY=<aws-secret-key> \
      -e AWS_REGION=<aws-region> \
      -e DATA_FILENAMES_CSV=<csv-filename> \
      -e DATA_REMOTE_PREFIX=<remote-prefix> \
      -e DATA_LOCAL_PREFIX=<destination-dir> \
      -e DATA_BUCKET_NAME=<s3-bucket> \
      fetch-s3  

Alternatively, the python script can take command-line parameters:

    python fetch-from-csv.py \
        --input <csv_filename> \
        --remote-prefix <remote-prefix> \
        --local-prefix <local-prefix> \
        --bucket-name <s3-bucket>

Help is available as:

    python fetch-from-csv.py --help

## Usage (Kubernetes)

This image is best used as an init-container, or the first task for a data processing stage.
