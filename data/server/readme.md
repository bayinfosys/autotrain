# Flask Video Server

Streams data from a directory on the host.

Set:
 + `DATA_PATH` environment variable to the target directory
 + `DATA_TITLE` environment variable to the title of the HTML page

## Usage

Example:

    DATA_PATH=/home/root/downloads python app.py

point your browser at http://127.0.0.1:8080

click some links

watch your videos

## Docker Usage

Mount your video directory to the `/data` in the container.

Example:

    docker build -t videoserver
    docker run --rm -p 80:8080 -v $(pwd)/video-data/:/data --name vidsrv videoserver


point your browser at http://127.0.0.1

click some links

watch your videos

## Kubernetes Usage
