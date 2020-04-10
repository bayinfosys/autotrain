import logging
from os.path import exists, join, dirname
from os import makedirs

logger = logging.getLogger(__name__)


def list_filenames_with_prefix(input_csv_filename, prefix):
  """list the files with the prefix prepended
  """
  import csv

  with open(input_csv_filename, "r") as csv_f:
    reader = csv.reader(csv_f)
    header = next(reader)
    filenames = [row for row in reader]

  if prefix is not None:
    filenames = [join(prefix, fname) for file_row in filenames for fname in file_row]

  return filenames


def download_objects(s3_session, bucket, remote_filenames, local_filenames):
  """download all the non-existing objects from the bucket
  """
  assert len(remote_filenames) == len(local_filenames)

  for remote, local in zip(remote_filenames, local_filenames):
    logger.info("downloading '%s':'%s' to '%s'" % (bucket, remote, local))
    if not exists(dirname(local)):
      makedirs(dirname(local))
    # FIXME: check md5?
    if not exists(local):
      s3_session.download_file(bucket, remote, local)


if __name__ == "__main__":
  import os
  import sys
  import argparse
  import boto3

  root = logging.getLogger()
  root.setLevel(logging.DEBUG)

  ch = logging.StreamHandler(sys.stdout)
  ch.setLevel(logging.INFO)
  formatter = logging.Formatter('%(name)s:%(levelname)s:%(message)s')
  ch.setFormatter(formatter)
  root.addHandler(ch)

  parser = argparse.ArgumentParser(
      description="download objects named in a csv file from s3")
  parser.add_argument(
      "-i",
      "--input",
      default=os.environ.get("DATA_FILENAMES_CSV", None),
      help="filename of csv holding download filenames",
  )
  parser.add_argument(
      "-r",
      "--remote-prefix",
      default=os.environ.get("DATA_REMOTE_PREFIX", None),
      help="URI for the remote system (e.g., https://my-bucket.s3.com)",
  )
  parser.add_argument(
      "-l",
      "--local-prefix",
      default=os.environ.get("DATA_LOCAL_PREFIX", None),
      help="prefix for the local system (e.g., /data/my-bucket/)",
  )
  parser.add_argument(
      "-b",
      "--bucket-name",
      default=os.environ.get("DATA_BUCKET_NAME", None),
      help="s3 bucket name",
  )

  args = parser.parse_args()

  assert args.input is not None
  assert args.remote_prefix is not None
  assert args.local_prefix is not None
  assert args.bucket_name is not None

  # create the session
  s3_session = boto3.client("s3")
  # download the files
  download_objects(s3_session,
                   args.bucket_name,
                   list_filenames_with_prefix(args.input, args.remote_prefix),
                   list_filenames_with_prefix(args.input, args.local_prefix))
