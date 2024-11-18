      
import sys
sys.path.append("C:\Users\timps\Documents\Queen's\ENPH 454\GitHub\ENPH454-TouchProjector\venv\Scripts\pymf.pyd")

import os
try:
    user_paths = os.environ['PYTHONPATH'].split(os.pathsep)
except KeyError:
    user_paths = []