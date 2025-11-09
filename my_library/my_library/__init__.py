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
