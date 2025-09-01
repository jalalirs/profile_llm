"""CLI module for ProfLLM"""

import sys
from pathlib import Path

# Add scripts directory to path and import main CLI
scripts_dir = Path(__file__).parent.parent / 'scripts'
sys.path.insert(0, str(scripts_dir))

from run import main

if __name__ == '__main__':
    main()