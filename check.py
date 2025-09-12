import sys
import platform
import pyspark

# Python version
print("Python version:", sys.version)
print("Python implementation:", platform.python_implementation())

# PySpark version
print("PySpark version:", pyspark.__version__)

# Java version (needed for Spark)
import subprocess
try:
    java_version = subprocess.check_output(["java", "-version"], stderr=subprocess.STDOUT)
    print(java_version.decode())
except Exception as e:
    print("Error getting Java version:", e)
