# 🎯 Market Fusion Analyzer

**Institutional-Grade Stock Market Analysis Library**  
*Combining Price Action + Open Interest Intelligence*

---

## 🚀 Overview

Market Fusion Analyzer is a Python library designed for algorithmic trading bots that need sophisticated market analysis by combining:
- **Multi-timeframe Price Action** (5m, 15m, 1h candles)
- **Open Interest Intelligence** (with time-based comparison)
- **Fusion Analysis** (reveals hidden market dynamics)

Perfect for Options Trading, Futures Trading, and Algorithmic Trading Bots.

---

## 📦 Installation

```bash
pip install -r requirements.txt
```

Or install directly:
```bash
pip install pandas numpy
```

---

## 🎨 Features

### 1️⃣ **Multi-Timeframe Price Action Analysis**
- Trend detection across 5-min, 15-min, and 1-hour timeframes
- Support & Resistance identification
- Pattern recognition
- Momentum analysis
- Alignment detection

### 2️⃣ **Open Interest Intelligence**
- PCR (Put-Call Ratio) tracking with historical comparison
- Max Pain calculation
- OI velocity analysis (change rate detection)
- OI wall detection (resistance/support zones)
- Smart money footprint identification

### 3️⃣ **Fusion Analysis** (The Secret Sauce 🔥)
- Combines Price Action + OI to detect:
  - Confirmed trends vs fake moves
  - Divergences (when price and OI disagree)
  - High-probability zones
  - Institutional activity patterns

### 4️⃣ **Trade Setup Generation**
- Entry points with OI confirmation
- Stop-loss based on price + OI invalidation
- Targets aligned with OI walls
- Confidence scoring

### 5️⃣ **Risk Assessment**
- Alignment strength rating
- Confidence levels
- Red flag detection

---

## 🛠️ Usage

### Basic Example

```python
from market_fusion import MarketFusionAnalyzer
from market_fusion.models.candle_data import Candle, MultiTimeframeData
from market_fusion.models.oi_data import StrikeOI, OISnapshot, OIComparison
from datetime import datetime

# Initialize
analyzer = MarketFusionAnalyzer()

# Prepare data (from your feed)
mtf_data = MultiTimeframeData(
    five_min=[...],      # List of Candle objects
    fifteen_min=[...],   # List of Candle objects
    one_hour=[...]       # List of Candle objects
)

oi_data = OIComparison(
    current=OISnapshot(...),
    fifteen_min_ago=OISnapshot(...),
    thirty_min_ago=OISnapshot(...)
)

# Run analysis
result = analyzer.analyze(
    symbol="NIFTY",
    current_price=44520.0,
    mtf_data=mtf_data,
    oi_data=oi_data,
    expiry_date="2025-11-15"
)

# Access results
print(result['fusion_analysis']['interpretation'])
print(result['trade_setups'])
print(result['risk_rating'])
```

---

## 📊 Output Structure

```python
{
    'symbol': 'NIFTY',
    'current_price': 44520.0,
    'timestamp': '2025-11-09T10:30:00',
    'expiry': '2025-11-15',
    
    'price_action': {
        'timeframes': {
            '5min': {'trend': 'bullish', 'strength': 75, 'momentum': 'accelerating'},
            '15min': {...},
            '1hour': {...}
        },
        'support_resistance': {
            'supports': [44400, 44350, 44300],
            'resistances': [44600, 44650, 44700]
        },
        'alignment': {
            'is_aligned': True,
            'dominant_trend': 'bullish',
            'strength': 78.5
        }
    },
    
    'oi_intelligence': {
        'pcr': {
            'current': 1.2,
            '15min_ago': 1.15,
            '30min_ago': 1.1,
            'trend': 'increasing'
        },
        'max_pain': 44500,
        'velocity': {
            'velocity_rating': 'fast',
            'is_accelerating': True
        },
        'walls': {
            'ce_walls': [(44600, 2500000), ...],
            'pe_walls': [(44400, 2800000), ...]
        }
    },
    
    'fusion_analysis': {
        'correlation': 'positive',
        'confidence': 'high',
        'max_pain_distance': 0.45,
        'interpretation': 'Strong bullish setup with OI confirmation...'
    },
    
    'trade_setups': [
        {
            'direction': 'LONG',
            'entry': 44520,
            'stop_loss': 44400,
            'target': 44600,
            'confidence': 'high',
            'reasoning': '...'
        }
    ],
    
    'risk_rating': {
        'rating': 'LOW_RISK',
        'score': 8.5,
        'confidence': 'high'
    }
}
```

---

## 🎯 Integration with Trading Bots

### Telegram Bot Example

```python
import time
from market_fusion import MarketFusionAnalyzer

analyzer = MarketFusionAnalyzer()

while True:
    # Fetch data from your feed
    candles = fetch_latest_candles()
    oi_data = fetch_option_chain()
    
    # Analyze
    result = analyzer.analyze(...)
    
    # Format for Telegram
    message = f"""
🎯 {result['symbol']} Analysis
Price: ₹{result['current_price']}

Trend: {result['fusion_analysis']['interpretation']}

Setup: {result['trade_setups'][0]['direction']}
Entry: {result['trade_setups'][0]['entry']}
SL: {result['trade_setups'][0]['stop_loss']}
Target: {result['trade_setups'][0]['target']}

Risk: {result['risk_rating']['rating']}
    """
    
    send_telegram_message(message)
    time.sleep(60)
```

---

## 📈 Key Concepts

### What is Fusion Analysis?

Traditional analysis looks at Price Action OR Open Interest separately. **Fusion Analysis** combines both to reveal:

1. **Confirmation**: When both agree → High confidence
2. **Divergence**: When they disagree → Fakeout warning
3. **Smart Money**: OI changes show where institutions are positioning
4. **Walls**: Heavy OI acts as magnetic support/resistance

### Example Scenarios

| Price Action | OI Change | Fusion Interpretation |
|--------------|-----------|----------------------|
| Bullish | CE OI increasing | ⚠️ Resistance building - possible rejection |
| Bullish | PE OI increasing | ✅ Bulls adding support - confirmed move |
| Bearish | CE OI decreasing | ✅ Resistance clearing - confirmed move |
| Bearish | PE OI decreasing | ⚠️ Support weakening - possible breakdown |

---

## 🔧 Advanced Features

### Custom Indicators

You can extend the library with your own indicators:

```python
from market_fusion.indicators import BaseIndicator

class MyCustomIndicator(BaseIndicator):
    def calculate(self, candles):
        # Your logic here
        return result
```

### Pattern Detection

```python
patterns = analyzer.price_analyzer.detect_patterns(candles)
# Returns: ['breakout', 'higher_highs', 'flag_pattern']
```

---

## 📝 Requirements

- Python 3.8+
- pandas >= 1.5.0
- numpy >= 1.23.0

---

## 🤝 Contributing

This library is designed to be extended. Feel free to:
- Add new indicators
- Improve pattern recognition
- Enhance OI analysis algorithms

---

## ⚠️ Disclaimer

This library is for educational and analysis purposes only. Always do your own research and risk management before trading.

---

## 📞 Support

For issues, feature requests, or questions, please open an issue on GitHub.

---

## 🎓 Learn More

For detailed documentation on each module, check:
- `docs/price_action.md` - Price Action Analysis
- `docs/oi_analysis.md` - OI Intelligence
- `docs/fusion.md` - Fusion Engine
- `docs/examples.md` - More Examples

---

**Made with ❤️ for Algo Traders**
