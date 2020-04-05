from pymongo import MongoClient
from bson.json_util import dumps
import os

print("'%s':'%s'" % (os.environ["MONGODB_USERNAME"], os.environ["MONGODB_HOSTNAME"]))

c = MongoClient("mongodb://%s" % (os.environ["MONGODB_HOSTNAME"]),
                username=os.environ["MONGODB_USERNAME"],
                password=os.environ["MONGODB_PASSWORD"])

print(dumps(c.admin.command("serverStatus")))
