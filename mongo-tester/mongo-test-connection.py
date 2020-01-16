from pymongo import MongoClient
import os

print("connecting to: '%s'" % os.environ["MONGODB_HOSTNAME"])
print("as user: '%s'" % os.environ["MONGODB_USERNAME"])

c = MongoClient("mongodb://%s" % (os.environ["MONGODB_HOSTNAME"]),
                username=os.environ["MONGODB_USERNAME"],
                password=os.environ["MONGODB_PASSWORD"])

c.admin.command("serverStatus")
