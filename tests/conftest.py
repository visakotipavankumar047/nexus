import os

os.environ["LANGSMITH_TRACING"] = "false"  # tests never send traces to the real project
