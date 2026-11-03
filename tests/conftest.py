import os
import tempfile
from pathlib import Path

# runs before the test modules are imported, so the API uses a throw-away database
os.environ["DATABASE_URL"] = "sqlite:///" + (Path(tempfile.mkdtemp()) / "test.db").as_posix()