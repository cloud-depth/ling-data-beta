import os
import sys
from src.processing_core import LingData
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

if __name__ == '__main__':
    LingData('databuilder_args/v0.0.3example.json').run_all()
