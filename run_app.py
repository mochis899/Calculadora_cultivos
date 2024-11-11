import os
import subprocess
import sys
from streamlit.web.cli import main

if __name__ == "__main__":
    sys.argv = ["streamlit", "run", os.path.join(os._MEIPASS if hasattr(sys, '_MEIPASS') else os.path.dirname(__file__), "app.py")]
    sys.exit(main())