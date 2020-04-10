import logging

from flask import (Flask,
                   render_template,
                   Response,
                   request,
                   jsonify,
                   make_response)

from flask_accept import accept, accept_fallback

import yaml

from os import listdir, stat, environ, getcwd
from os.path import join, isfile

logger = logging.getLogger()

# 10Mb buffer
BUFF_SIZE=10*(1 << 20)
DATA_PATH=environ.get("DATA_PATH", getcwd())
TITLE=environ.get("DATA_TITLE", getcwd())

app = Flask(__name__)

def get_file_list():
  """get a list of the files in our path
  """
  # FIXME: get subdirs?
  # FIXME: filter by file type
  logger.info("enumerating: '%s'" % str(DATA_PATH))
  file_list = [f for f in listdir(DATA_PATH) if isfile(join(DATA_PATH, f))]
  logger.info("found %i files" % len(file_list))
  return file_list


@app.route('/')
@accept_fallback
@accept('text/html')
def index():
  """list files in the dir
  """
  return render_template("index.html",
                         data={"title": environ.get("TITLE", getcwd()),
                               "files": get_file_list()})

@index.support('application/json')
def index_json():
  return jsonify({"title": environ.get("TITLE", getcwd()),
                  "files": get_file_list()})

@index.support('application/yaml')
def index_yaml():
  resp = make_response(yaml.dump({"title": environ.get("TITLE", getcwd()),
                                  "files": get_file_list()}), 200)
  resp.headers["Content-Type"] = "application/yaml"
  return resp


def get_data(file_path, data_start, data_length=BUFF_SIZE):
  """get some data from disk
  """
  # read some info about the file
  file_length = stat(join(VIDEO_PATH, file_path)).st_size
  file_start = 0

  data_start = min(data_start, file_start+file_length)
  data_length = min(data_length, (file_start+file_length)-data_start)

  with open(join(VIDEO_PATH, file_path), 'rb') as f:
    f.seek(data_start)
    data = f.read(data_length)

  return data, data_start, data_length, file_length


@app.route('/data/<filename>')
def get_file(filename):
  """stream the file
  """
  import re

  range_start = 0
  range_length = BUFF_SIZE

  if request.headers.has_key("Range"):
    range_header = request.headers.get('Range', None)

    match = re.search(r'(\d+)-(\d*)', request.headers["Range"])
    groups = match.groups()

    if groups[0]:
      range_start = int(groups[0])
    if groups[1]:
      range_length = int(groups[1])

  data, start, length, file_size = get_data(filename, range_start, range_length)

  # FIXME: adjust the mimetype for the filetype
  resp = Response(data,
                  206,
                  mimetype="video/mp4",
                  content_type="video/mp4",
                  direct_passthrough=True)

  resp.headers.add("Content-Range",
                   "bytes %i-%i/%i" % (start, start+length-1, file_size))
  resp.headers.add("Accept-Ranges", "bytes")

  return resp


if __name__ == "__main__":
  logging.basicConfig(level=logging.INFO)
  app.run(host="0.0.0.0",
          port=environ("PORT", 80),
          debug=environ("DEBUG", False),
          threaded=True)
