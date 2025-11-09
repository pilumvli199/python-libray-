# ═══════════════════════════════════════════════════════════════════════════════
# FILE 1: __init__.py
# ═══════════════════════════════════════════════════════════════════════════════
"""
Market Fusion Analyzer
======================

A Python library for institutional-grade stock market analysis combining 
Price Action and Open Interest Intelligence.

Features:
---------
- Multi-timeframe Price Action Analysis (5m, 15m, 1h)
- Open Interest Intelligence with time-based comparison
- Fusion Analysis combining Price + OI
- Candlestick Pattern Recognition
- Trade Setup Generation with Risk Assessment

Usage:
------
>>> from market_fusion import MarketFusionAnalyzer, format_output_for_bot
>>> 
>>> analyzer = MarketFusionAnalyzer()
>>> result = analyzer.analyze_from_json(json_data)
>>> print(format_output_for_bot(result))

Author: Your Name
Version: 1.0.0
License: MIT
"""

__version__ = "1.0.0"
__author__ = "Your Name"
__license__ = "MIT"
__email__ = "your.email@example.com"

# Import main classes for easy access
from main import (
    MarketFusionAnalyzer,
    PriceActionAnalyzer,
    OIAnalyzer,
    CandleData,
    OIData,
    format_output_for_bot
)

# Define what gets imported with "from market_fusion import *"
__all__ = [
    'MarketFusionAnalyzer',
    'PriceActionAnalyzer',
    'OIAnalyzer',
    'CandleData',
    'OIData',
    'format_output_for_bot',
    '__version__'
]


# ═══════════════════════════════════════════════════════════════════════════════
# FILE 2: setup.py
# ═══════════════════════════════════════════════════════════════════════════════

from setuptools import setup, find_packages
import os

# Read README for long description
def read_file(filename):
    """Read file contents"""
    filepath = os.path.join(os.path.dirname(__file__), filename)
    if os.path.exists(filepath):
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read()
    return ''

# Read requirements
def read_requirements():
    """Read requirements from requirements.txt"""
    filepath = os.path.join(os.path.dirname(__file__), 'requirements.txt')
    if os.path.exists(filepath):
        with open(filepath, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f if line.strip() and not line.startswith('#')]
    return ['pandas>=2.0.0', 'numpy>=1.24.0']

setup(
    # Basic Package Info
    name='market-fusion-analyzer',
    version='1.0.0',
    description='Institutional-grade Stock Market Analysis Library combining Price Action and OI Intelligence',
    long_description=read_file('README.md'),
    long_description_content_type='text/markdown',
    
    # Author Details
    author='Your Name',
    author_email='your.email@example.com',
    url='https://github.com/yourusername/market-fusion',
    
    # License
    license='MIT',
    
    # Package Discovery
    packages=find_packages(exclude=['tests', 'docs', 'examples']),
    py_modules=['main'],  # Include main.py as a module
    
    # Dependencies
    install_requires=read_requirements(),
    
    # Python Version Requirement
    python_requires='>=3.8',
    
    # Additional Requirements (Optional)
    extras_require={
        'dev': [
            'pytest>=7.0.0',
            'pytest-cov>=3.0.0',
            'black>=22.0.0',
            'flake8>=4.0.0',
        ],
        'docs': [
            'sphinx>=4.0.0',
            'sphinx-rtd-theme>=1.0.0',
        ]
    },
    
    # Package Data
    include_package_data=True,
    package_data={
        '': ['*.txt', '*.md', '*.rst'],
    },
    
    # Classification
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Financial and Insurance Industry',
        'Topic :: Office/Business :: Financial :: Investment',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Operating System :: OS Independent',
    ],
    
    # Keywords for PyPI search
    keywords=[
        'stock market',
        'trading',
        'technical analysis',
        'price action',
        'open interest',
        'options trading',
        'algorithmic trading',
        'market analysis',
        'nifty',
        'banknifty',
        'indian stock market'
    ],
    
    # Entry Points (CLI commands if needed)
    entry_points={
        'console_scripts': [
            'market-fusion=main:example_usage',  # CLI command: market-fusion
        ],
    },
    
    # Project URLs
    project_urls={
        'Bug Reports': 'https://github.com/yourusername/market-fusion/issues',
        'Source': 'https://github.com/yourusername/market-fusion',
        'Documentation': 'https://github.com/yourusername/market-fusion/wiki',
    },
    
    # Zip Safe
    zip_safe=False,
)


# ═══════════════════════════════════════════════════════════════════════════════
# FILE 3: MANIFEST.in (Optional - for including non-Python files)
# ═══════════════════════════════════════════════════════════════════════════════
"""
include README.md
include LICENSE
include requirements.txt
recursive-include docs *.md *.rst
recursive-include examples *.py
recursive-include tests *.py
global-exclude __pycache__
global-exclude *.py[co]
"""


# ═══════════════════════════════════════════════════════════════════════════════
# FILE 4: .gitignore (Recommended)
# ═══════════════════════════════════════════════════════════════════════════════
"""
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Virtual Environment
venv/
env/
ENV/

# IDEs
.vscode/
.idea/
*.swp
*.swo
*~

# Testing
.pytest_cache/
.coverage
htmlcov/

# OS
.DS_Store
Thumbs.db
"""


# ═══════════════════════════════════════════════════════════════════════════════
# INSTALLATION INSTRUCTIONS
# ═══════════════════════════════════════════════════════════════════════════════
"""
════════════════════════════════════════════════════════════════════════════════
INSTALLATION METHODS
════════════════════════════════════════════════════════════════════════════════

METHOD 1: Local Installation (Development)
────────────────────────────────────────────
# Clone repository
git clone https://github.com/yourusername/market-fusion.git
cd market-fusion

# Install in editable mode
pip install -e .

# Or install directly
pip install .


METHOD 2: Install from PyPI (After Publishing)
───────────────────────────────────────────────
pip install market-fusion-analyzer


METHOD 3: Install from GitHub
──────────────────────────────
pip install git+https://github.com/yourusername/market-fusion.git


METHOD 4: Install with extras
──────────────────────────────
# With development tools
pip install market-fusion-analyzer[dev]

# With documentation tools
pip install market-fusion-analyzer[docs]

# With all extras
pip install market-fusion-analyzer[dev,docs]


════════════════════════════════════════════════════════════════════════════════
USAGE AFTER INSTALLATION
════════════════════════════════════════════════════════════════════════════════

Method 1: As Python Module
───────────────────────────
from market_fusion import MarketFusionAnalyzer, format_output_for_bot

analyzer = MarketFusionAnalyzer()
result = analyzer.analyze_from_json(json_data)
print(format_output_for_bot(result))


Method 2: As CLI Command
─────────────────────────
# Run example analysis
market-fusion


════════════════════════════════════════════════════════════════════════════════
PUBLISHING TO PyPI (Optional)
════════════════════════════════════════════════════════════════════════════════

Step 1: Create PyPI Account
────────────────────────────
Visit: https://pypi.org/account/register/


Step 2: Install Publishing Tools
─────────────────────────────────
pip install twine build


Step 3: Build Package
─────────────────────
python -m build


Step 4: Upload to TestPyPI (Testing)
─────────────────────────────────────
twine upload --repository testpypi dist/*


Step 5: Upload to PyPI (Production)
────────────────────────────────────
twine upload dist/*


════════════════════════════════════════════════════════════════════════════════
"""
