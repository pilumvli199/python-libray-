"""
═══════════════════════════════════════════════════════════════════════════════
MARKET FUSION ANALYZER - Complete Single File Library
Price Action + OI Intelligence Combined Analysis
═══════════════════════════════════════════════════════════════════════════════
"""

import json
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Any, Optional


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1: DATA MODELS
# ═══════════════════════════════════════════════════════════════════════════════

class CandleData:
    """Handles candle data parsing from JSON"""
    
    @staticmethod
    def from_json(json_data: List[Dict]) -> pd.DataFrame:
        """
        Converts JSON candle data to DataFrame
        
        Expected JSON format:
        [
            {
                "timestamp": "2025-11-09 10:30:00",
                "open": 44500.0,
                "high": 44550.0,
                "low": 44450.0,
                "close": 44520.0,
                "volume": 150000
            },
            ...
        ]
        """
        df = pd.DataFrame(json_data)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp')
        return df


class OIData:
    """Handles OI data parsing from JSON"""
    
    @staticmethod
    def from_json(json_data: Dict) -> Dict:
        """
        Converts JSON OI data to structured format
        
        Expected JSON format:
        {
            "current": {
                "timestamp": "2025-11-09 10:30:00",
                "strikes": [
                    {"strike": 44000, "ce_oi": 1500000, "pe_oi": 800000, "ce_volume": 50000, "pe_volume": 30000},
                    {"strike": 44100, "ce_oi": 1400000, "pe_oi": 900000, ...},
                    ...
                ]
            },
            "fifteen_min_ago": {...},
            "thirty_min_ago": {...}
        }
        """
        return {
            'current': pd.DataFrame(json_data['current']['strikes']),
            'fifteen_min_ago': pd.DataFrame(json_data['fifteen_min_ago']['strikes']),
            'thirty_min_ago': pd.DataFrame(json_data['thirty_min_ago']['strikes'])
        }


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2: PRICE ACTION ANALYZER
# ═══════════════════════════════════════════════════════════════════════════════

class PriceActionAnalyzer:
    """Analyzes price action and identifies patterns"""
    
    def __init__(self):
        self.patterns = []
    
    def analyze_trend(self, df: pd.DataFrame, timeframe: str) -> Dict:
        """Analyzes trend from candle DataFrame"""
        
        if len(df) < 50:
            return {'trend': 'neutral', 'strength': 0, 'momentum': 'unknown', 'timeframe': timeframe}
        
        # Calculate EMAs
        df['ema20'] = df['close'].ewm(span=20, adjust=False).mean()
        df['ema50'] = df['close'].ewm(span=50, adjust=False).mean()
        
        current_close = df['close'].iloc[-1]
        ema20 = df['ema20'].iloc[-1]
        ema50 = df['ema50'].iloc[-1]
        
        # Determine trend
        if current_close > ema20 > ema50:
            trend = 'bullish'
            strength = min(100, ((current_close - ema50) / ema50) * 1000)
        elif current_close < ema20 < ema50:
            trend = 'bearish'
            strength = min(100, ((ema50 - current_close) / ema50) * 1000)
        else:
            trend = 'neutral'
            strength = 50
        
        # Calculate momentum
        if len(df) >= 15:
            recent_momentum = df['close'].pct_change().tail(5).mean()
            older_momentum = df['close'].pct_change().tail(10).head(5).mean()
            momentum = 'accelerating' if abs(recent_momentum) > abs(older_momentum) else 'decelerating'
        else:
            momentum = 'neutral'
        
        return {
            'trend': trend,
            'strength': round(float(strength), 2),
            'momentum': momentum,
            'timeframe': timeframe,
            'ema20': round(float(ema20), 2),
            'ema50': round(float(ema50), 2)
        }
    
    def find_support_resistance(self, df: pd.DataFrame) -> Dict:
        """Finds key support and resistance levels"""
        
        if len(df) < 20:
            return {'supports': [], 'resistances': []}
        
        # Find pivot highs and lows
        window = 5
        highs = df['high'].values
        lows = df['low'].values
        
        pivot_highs = []
        pivot_lows = []
        
        for i in range(window, len(df) - window):
            # Check for pivot high
            if highs[i] == max(highs[i-window:i+window+1]):
                pivot_highs.append(highs[i])
            
            # Check for pivot low
            if lows[i] == min(lows[i-window:i+window+1]):
                pivot_lows.append(lows[i])
        
        # Get top 3 most significant levels
        resistances = sorted(set(pivot_highs), reverse=True)[:3]
        supports = sorted(set(pivot_lows))[:3]
        
        return {
            'supports': [round(float(s), 2) for s in supports],
            'resistances': [round(float(r), 2) for r in resistances]
        }
    
    def detect_candlestick_patterns(self, df: pd.DataFrame) -> List[str]:
        """Detects candlestick patterns in recent candles"""
        
        if len(df) < 3:
            return []
        
        patterns = []
        
        # Get last 3 candles
        c1 = df.iloc[-3]
        c2 = df.iloc[-2]
        c3 = df.iloc[-1]
        
        # Calculate body and wick sizes
        body3 = abs(c3['close'] - c3['open'])
        upper_wick3 = c3['high'] - max(c3['open'], c3['close'])
        lower_wick3 = min(c3['open'], c3['close']) - c3['low']
        
        # Doji pattern
        if body3 < (c3['high'] - c3['low']) * 0.1:
            patterns.append('DOJI')
        
        # Hammer pattern (bullish)
        if lower_wick3 > body3 * 2 and upper_wick3 < body3 * 0.5 and c3['close'] < c3['open']:
            patterns.append('HAMMER')
        
        # Shooting Star (bearish)
        if upper_wick3 > body3 * 2 and lower_wick3 < body3 * 0.5 and c3['close'] < c3['open']:
            patterns.append('SHOOTING_STAR')
        
        # Bullish Engulfing
        if (c2['close'] < c2['open'] and  # Previous bearish
            c3['close'] > c3['open'] and  # Current bullish
            c3['open'] < c2['close'] and  # Opens below previous close
            c3['close'] > c2['open']):    # Closes above previous open
            patterns.append('BULLISH_ENGULFING')
        
        # Bearish Engulfing
        if (c2['close'] > c2['open'] and  # Previous bullish
            c3['close'] < c3['open'] and  # Current bearish
            c3['open'] > c2['close'] and  # Opens above previous close
            c3['close'] < c2['open']):    # Closes below previous open
            patterns.append('BEARISH_ENGULFING')
        
        # Three White Soldiers (bullish)
        if (c1['close'] > c1['open'] and
            c2['close'] > c2['open'] and
            c3['close'] > c3['open'] and
            c2['close'] > c1['close'] and
            c3['close'] > c2['close']):
            patterns.append('THREE_WHITE_SOLDIERS')
        
        # Three Black Crows (bearish)
        if (c1['close'] < c1['open'] and
            c2['close'] < c2['open'] and
            c3['close'] < c3['open'] and
            c2['close'] < c1['close'] and
            c3['close'] < c2['close']):
            patterns.append('THREE_BLACK_CROWS')
        
        return patterns
    
    def calculate_momentum_indicators(self, df: pd.DataFrame) -> Dict:
        """Calculates RSI and other momentum indicators"""
        
        if len(df) < 14:
            return {'rsi': 50, 'macd': 0, 'signal': 0}
        
        # RSI Calculation
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        # MACD Calculation
        ema12 = df['close'].ewm(span=12, adjust=False).mean()
        ema26 = df['close'].ewm(span=26, adjust=False).mean()
        macd = ema12 - ema26
        signal = macd.ewm(span=9, adjust=False).mean()
        
        return {
            'rsi': round(float(rsi.iloc[-1]), 2),
            'macd': round(float(macd.iloc[-1]), 2),
            'signal': round(float(signal.iloc[-1]), 2),
            'macd_histogram': round(float(macd.iloc[-1] - signal.iloc[-1]), 2)
        }


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3: OI ANALYZER
# ═══════════════════════════════════════════════════════════════════════════════

class OIAnalyzer:
    """Analyzes Open Interest data"""
    
    def calculate_pcr(self, df: pd.DataFrame) -> float:
        """Calculates Put-Call Ratio"""
        total_pe_oi = df['pe_oi'].sum()
        total_ce_oi = df['ce_oi'].sum()
        return round(float(total_pe_oi / total_ce_oi), 3) if total_ce_oi > 0 else 0
    
    def calculate_max_pain(self, df: pd.DataFrame) -> float:
        """Calculates Max Pain strike"""
        pain_values = {}
        
        for _, strike_row in df.iterrows():
            strike = strike_row['strike']
            pain = 0
            
            for _, s in df.iterrows():
                if s['strike'] > strike:
                    pain += (s['strike'] - strike) * s['ce_oi']
                elif s['strike'] < strike:
                    pain += (strike - s['strike']) * s['pe_oi']
            
            pain_values[strike] = pain
        
        if not pain_values:
            return 0
        
        max_pain_strike = min(pain_values, key=pain_values.get)
        return float(max_pain_strike)
    
    def analyze_oi_changes(self, current_df: pd.DataFrame, 
                          prev_15_df: pd.DataFrame, 
                          prev_30_df: pd.DataFrame) -> Dict:
        """Analyzes OI changes over time"""
        
        # Merge dataframes on strike
        merged_15 = current_df.merge(prev_15_df, on='strike', suffixes=('_now', '_15min'))
        merged_30 = current_df.merge(prev_30_df, on='strike', suffixes=('_now', '_30min'))
        
        # Calculate changes
        ce_change_15 = (merged_15['ce_oi_now'] - merged_15['ce_oi_15min']).sum()
        pe_change_15 = (merged_15['pe_oi_now'] - merged_15['pe_oi_15min']).sum()
        
        ce_change_30 = (merged_30['ce_oi_now'] - merged_30['ce_oi_30min']).sum()
        pe_change_30 = (merged_30['pe_oi_now'] - merged_30['pe_oi_30min']).sum()
        
        return {
            'ce_change_15min': int(ce_change_15),
            'pe_change_15min': int(pe_change_15),
            'ce_change_30min': int(ce_change_30),
            'pe_change_30min': int(pe_change_30),
            'total_change_15min': int(ce_change_15 + pe_change_15),
            'total_change_30min': int(ce_change_30 + pe_change_30)
        }
    
    def find_oi_walls(self, df: pd.DataFrame, current_price: float) -> Dict:
        """Finds significant OI concentration (walls)"""
        
        avg_ce_oi = df['ce_oi'].mean()
        avg_pe_oi = df['pe_oi'].mean()
        
        # Find CE walls (resistance above current price)
        ce_walls = df[
            (df['ce_oi'] > avg_ce_oi * 1.5) & 
            (df['strike'] > current_price)
        ].nlargest(3, 'ce_oi')[['strike', 'ce_oi']].to_dict('records')
        
        # Find PE walls (support below current price)
        pe_walls = df[
            (df['pe_oi'] > avg_pe_oi * 1.5) & 
            (df['strike'] < current_price)
        ].nlargest(3, 'pe_oi')[['strike', 'pe_oi']].to_dict('records')
        
        return {
            'ce_walls': ce_walls,
            'pe_walls': pe_walls
        }
    
    def detect_oi_buildup(self, current_df: pd.DataFrame, 
                          prev_df: pd.DataFrame, 
                          current_price: float) -> Dict:
        """Detects OI buildup patterns"""
        
        merged = current_df.merge(prev_df, on='strike', suffixes=('_now', '_prev'))
        
        # Long buildup: CE OI increasing + PE OI increasing (consolidation)
        # Short buildup: CE OI decreasing + PE OI decreasing (consolidation)
        # Long unwinding: CE OI decreasing (bearish)
        # Short covering: PE OI decreasing (bullish)
        
        atm_strike = merged.iloc[(merged['strike'] - current_price).abs().argsort()[:1]]
        
        if atm_strike.empty:
            return {'pattern': 'unknown', 'signal': 'neutral'}
        
        ce_change = atm_strike['ce_oi_now'].values[0] - atm_strike['ce_oi_prev'].values[0]
        pe_change = atm_strike['pe_oi_now'].values[0] - atm_strike['pe_oi_prev'].values[0]
        
        if ce_change > 0 and pe_change > 0:
            return {'pattern': 'LONG_BUILDUP', 'signal': 'bullish'}
        elif ce_change < 0 and pe_change < 0:
            return {'pattern': 'SHORT_BUILDUP', 'signal': 'bearish'}
        elif ce_change < 0 and pe_change > 0:
            return {'pattern': 'LONG_UNWINDING', 'signal': 'bearish'}
        elif ce_change > 0 and pe_change < 0:
            return {'pattern': 'SHORT_COVERING', 'signal': 'bullish'}
        else:
            return {'pattern': 'MIXED', 'signal': 'neutral'}


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4: FUSION ENGINE (MAIN ANALYZER)
# ═══════════════════════════════════════════════════════════════════════════════

class MarketFusionAnalyzer:
    """
    Main engine combining Price Action + OI Analysis
    """
    
    def __init__(self):
        self.price_analyzer = PriceActionAnalyzer()
        self.oi_analyzer = OIAnalyzer()
    
    def analyze_from_json(self, json_input: str) -> Dict[str, Any]:
        """
        Main analysis function - accepts JSON string input
        
        Expected JSON structure:
        {
            "symbol": "NIFTY",
            "current_price": 44520.0,
            "expiry_date": "2025-11-15",
            "candles": {
                "5min": [...],
                "15min": [...],
                "1hour": [...]
            },
            "oi_data": {
                "current": {...},
                "fifteen_min_ago": {...},
                "thirty_min_ago": {...}
            }
        }
        """
        
        # Parse JSON
        data = json.loads(json_input) if isinstance(json_input, str) else json_input
        
        # Extract data
        symbol = data['symbol']
        current_price = data['current_price']
        expiry_date = data.get('expiry_date', 'N/A')
        
        # Convert candles to DataFrames
        candles_5m = CandleData.from_json(data['candles']['5min'])
        candles_15m = CandleData.from_json(data['candles']['15min'])
        candles_1h = CandleData.from_json(data['candles']['1hour'])
        
        # Convert OI data
        oi_dfs = OIData.from_json(data['oi_data'])
        
        # Perform analysis
        return self.analyze(
            symbol=symbol,
            current_price=current_price,
            expiry_date=expiry_date,
            candles_5m=candles_5m,
            candles_15m=candles_15m,
            candles_1h=candles_1h,
            oi_current=oi_dfs['current'],
            oi_15min=oi_dfs['fifteen_min_ago'],
            oi_30min=oi_dfs['thirty_min_ago']
        )
    
    def analyze(self, symbol: str, current_price: float, expiry_date: str,
                candles_5m: pd.DataFrame, candles_15m: pd.DataFrame, 
                candles_1h: pd.DataFrame, oi_current: pd.DataFrame,
                oi_15min: pd.DataFrame, oi_30min: pd.DataFrame) -> Dict[str, Any]:
        """
        Complete analysis combining Price Action + OI
        """
        
        # 1. Price Action Analysis
        price_5m = self.price_analyzer.analyze_trend(candles_5m, '5min')
        price_15m = self.price_analyzer.analyze_trend(candles_15m, '15min')
        price_1h = self.price_analyzer.analyze_trend(candles_1h, '1hour')
        
        sr_levels = self.price_analyzer.find_support_resistance(candles_1h)
        patterns = self.price_analyzer.detect_candlestick_patterns(candles_1h)
        momentum = self.price_analyzer.calculate_momentum_indicators(candles_1h)
        
        # 2. OI Analysis
        pcr_current = self.oi_analyzer.calculate_pcr(oi_current)
        pcr_15min = self.oi_analyzer.calculate_pcr(oi_15min)
        pcr_30min = self.oi_analyzer.calculate_pcr(oi_30min)
        
        max_pain = self.oi_analyzer.calculate_max_pain(oi_current)
        oi_changes = self.oi_analyzer.analyze_oi_changes(oi_current, oi_15min, oi_30min)
        oi_walls = self.oi_analyzer.find_oi_walls(oi_current, current_price)
        oi_buildup = self.oi_analyzer.detect_oi_buildup(oi_current, oi_15min, current_price)
        
        # 3. Alignment Check
        trends = [price_5m['trend'], price_15m['trend'], price_1h['trend']]
        aligned = len(set(trends)) == 1
        dominant_trend = max(set(trends), key=trends.count)
        
        # 4. Fusion Analysis
        fusion = self._fuse_analysis(
            dominant_trend, pcr_current, pcr_15min, 
            max_pain, current_price, oi_buildup
        )
        
        # 5. Generate Trade Setups
        trade_setups = self._generate_setups(
            fusion, dominant_trend, current_price, 
            sr_levels, oi_walls, momentum
        )
        
        # 6. Risk Rating
        risk = self._calculate_risk(fusion, aligned, momentum)
        
        return {
            'symbol': symbol,
            'current_price': current_price,
            'timestamp': datetime.now().isoformat(),
            'expiry': expiry_date,
            
            'price_action': {
                'timeframes': {
                    '5min': price_5m,
                    '15min': price_15m,
                    '1hour': price_1h
                },
                'support_resistance': sr_levels,
                'patterns': patterns,
                'momentum': momentum,
                'alignment': {
                    'is_aligned': aligned,
                    'dominant_trend': dominant_trend,
                    'agreement': f"{trends.count(dominant_trend)}/3 timeframes"
                }
            },
            
            'oi_intelligence': {
                'pcr': {
                    'current': pcr_current,
                    '15min_ago': pcr_15min,
                    '30min_ago': pcr_30min,
                    'trend': 'increasing' if pcr_current > pcr_15min else 'decreasing'
                },
                'max_pain': max_pain,
                'max_pain_distance': round(((current_price - max_pain) / max_pain) * 100, 2),
                'oi_changes': oi_changes,
                'oi_walls': oi_walls,
                'oi_buildup': oi_buildup
            },
            
            'fusion_analysis': fusion,
            'trade_setups': trade_setups,
            'risk_rating': risk
        }
    
    def _fuse_analysis(self, pa_trend: str, pcr_now: float, pcr_prev: float,
                      max_pain: float, current_price: float, oi_buildup: Dict) -> Dict:
        """Combines Price Action + OI for fusion insights"""
        
        pcr_trend = 'increasing' if pcr_now > pcr_prev else 'decreasing'
        distance_from_mp = ((current_price - max_pain) / max_pain) * 100
        
        # Determine correlation
        if pa_trend == 'bullish' and pcr_trend == 'decreasing':
            correlation = 'POSITIVE'
            confidence = 'HIGH'
        elif pa_trend == 'bearish' and pcr_trend == 'increasing':
            correlation = 'POSITIVE'
            confidence = 'HIGH'
        elif pa_trend == 'bullish' and pcr_trend == 'increasing':
            correlation = 'DIVERGENCE'
            confidence = 'LOW'
        elif pa_trend == 'bearish' and pcr_trend == 'decreasing':
            correlation = 'DIVERGENCE'
            confidence = 'LOW'
        else:
            correlation = 'NEUTRAL'
            confidence = 'MEDIUM'
        
        # Generate interpretation
        interpretation = self._interpret_fusion(
            pa_trend, correlation, distance_from_mp, 
            oi_buildup['pattern'], oi_buildup['signal']
        )
        
        return {
            'correlation': correlation,
            'confidence': confidence,
            'max_pain_distance_pct': round(distance_from_mp, 2),
            'oi_buildup_pattern': oi_buildup['pattern'],
            'oi_signal': oi_buildup['signal'],
            'interpretation': interpretation
        }
    
    def _interpret_fusion(self, pa_trend: str, correlation: str, 
                         mp_distance: float, oi_pattern: str, oi_signal: str) -> str:
        """Generates human-readable interpretation"""
        
        if correlation == 'POSITIVE':
            if abs(mp_distance) < 2:
                return f"🎯 Strong {pa_trend} setup with OI confirmation. Price near max pain - high probability zone. OI shows {oi_pattern}."
            else:
                return f"✅ {pa_trend.upper()} trend confirmed by OI data. {oi_pattern} detected. Good alignment."
        
        elif correlation == 'DIVERGENCE':
            return f"⚠️ CAUTION: Price shows {pa_trend} but OI suggests opposite ({oi_signal}). Possible fakeout risk. Wait for confirmation."
        
        else:
            return f"⏸️ Mixed signals. Price {pa_trend}, OI showing {oi_pattern}. Wait for clearer direction."
    
    def _generate_setups(self, fusion: Dict, trend: str, current_price: float,
                        sr_levels: Dict, oi_walls: Dict, momentum: Dict) -> List[Dict]:
        """Generates actionable trade setups"""
        
        setups = []
        
        # Only generate setup if confidence is medium or high
        if fusion['confidence'] in ['HIGH', 'MEDIUM']:
            
            if trend == 'bullish' and fusion['oi_signal'] == 'bullish':
                setup = {
                    'direction': 'LONG',
                    'entry_price': current_price,
                    'stop_loss': sr_levels['supports'][0] if sr_levels['supports'] else current_price * 0.98,
                    'target': sr_levels['resistances'][0] if sr_levels['resistances'] else current_price * 1.02,
                    'confidence': fusion['confidence'],
                    'reasoning': fusion['interpretation'],
                    'rsi': momentum['rsi'],
                    'risk_reward': 'calculating...'
                }
                
                # Calculate risk-reward
                risk = setup['entry_price'] - setup['stop_loss']
                reward = setup['target'] - setup['entry_price']
                setup['risk_reward'] = f"1:{round(reward/risk, 2)}" if risk > 0 else 'N/A'
                
                setups.append(setup)
            
            elif trend == 'bearish' and fusion['oi_signal'] == 'bearish':
                setup = {
                    'direction': 'SHORT',
                    'entry_price': current_price,
                    'stop_loss': sr_levels['resistances'][0] if sr_levels['resistances'] else current_price * 1.02,
                    'target': sr_levels['supports'][0] if sr_levels['supports'] else current_price * 0.98,
                    'confidence': fusion['confidence'],
                    'reasoning': fusion['interpretation'],
                    'rsi': momentum['rsi'],
                    'risk_reward': 'calculating...'
                }
                
                risk = setup['stop_loss'] - setup['entry_price']
                reward = setup['entry_price'] - setup['target']
                setup['risk_reward'] = f"1:{round(reward/risk, 2)}" if risk > 0 else 'N/A'
                
                setups.append(setup)
        
        return setups
    
    def _calculate_risk(self, fusion: Dict, aligned: bool, momentum: Dict) -> Dict:
        """Calculates overall risk rating"""
        
        score = 0
        
        # Confidence score
        if fusion['confidence'] == 'HIGH':
            score += 4
        elif fusion['confidence'] == 'MEDIUM':
            score += 2
        
        # Alignment score
        if aligned:
            score += 3
        
        # RSI score (neutral RSI is safer)
        rsi = momentum['rsi']
        if 40 <= rsi <= 60:
            score += 3
        elif 30 <= rsi <= 70:
            score += 1
        
        # Total score out of 10
        if score >= 8:
            rating = 'LOW_RISK'
        elif score >= 5:
            rating = 'MEDIUM_RISK'
        else:
            rating = 'HIGH_RISK'
        
        return {
            'rating': rating,
            'score': score,
            'max_score': 10,
            'confidence': fusion['confidence']
        }


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 5: HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def format_output_for_bot(result: Dict) -> str:
    """Formats analysis output for bot display"""
    
    msg = f"""
╔══════════════════════════════════════╗
║   🎯 MARKET FUSION ANALYSIS 🎯      ║
╚══════════════════════════════════════╝

📊 **{result['symbol']}** | ₹{result['current_price']}
⏰ {result['timestamp'][:19]}
📅 Expiry: {result['expiry']}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📈 **PRICE ACTION**
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
5m  : {result['price_action']['timeframes']['5min']['trend'].upper()} 
      (Strength: {result['price_action']['timeframes']['5min']['strength']}%)
15m : {result['price_action']['timeframes']['15min']['trend'].upper()} 
      (Strength: {result['price_action']['timeframes']['15min']['strength']}%)
1h  : {result['price_action']['timeframes']['1hour']['trend'].upper()} 
      (Strength: {result['price_action']['timeframes']['1hour']['strength']}%)

📍 Alignment: {'✅ YES' if result['price_action']['alignment']['is_aligned'] else '❌ NO'}
🎯 Dominant: {result['price_action']['alignment']['dominant_trend'].upper()}

📊 RSI: {result['price_action']['momentum']['rsi']} 
   MACD: {result['price_action']['momentum']['macd']}

🔺 Resistance: {', '.join(map(str, result['price_action']['support_resistance']['resistances'][:2]))}
🔻 Support: {', '.join(map(str, result['price_action']['support_resistance']['supports'][:2]))}

🕯️ Patterns: {', '.join(result['price_action']['patterns']) if result['price_action']['patterns'] else 'None detected'}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🔥 **OI INTELLIGENCE**
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PCR Now    : {result['oi_intelligence']['pcr']['current']}
PCR 15min  : {result['oi_intelligence']['pcr']['15min_ago']}
PCR Trend  : {result['oi_intelligence']['pcr']['trend'].upper()}

💰 Max Pain: {result['oi_intelligence']['max_pain']}
📏 Distance: {result['oi_intelligence']['max_pain_distance']}%

📈 OI Changes (15min):
   CE: {result['oi_intelligence']['oi_changes']['ce_change_15min']:,}
   PE: {result['oi_intelligence']['oi_changes']['pe_change_15min']:,}

🧱 OI Pattern: {result['oi_intelligence']['oi_buildup']['pattern']}
🎯 OI Signal: {result['oi_intelligence']['oi_buildup']['signal'].upper()}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
⚡ **FUSION ANALYSIS**
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Correlation: {result['fusion_analysis']['correlation']}
Confidence: {result['fusion_analysis']['confidence']}

💡 {result['fusion_analysis']['interpretation']}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
💼 **TRADE SETUPS**
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
    
    if result['trade_setups']:
        for i, setup in enumerate(result['trade_setups'], 1):
            msg += f"""
Setup #{i}: {setup['direction']}
├─ Entry: ₹{setup['entry_price']}
├─ SL: ₹{setup['stop_loss']}
├─ Target: ₹{setup['target']}
├─ R:R: {setup['risk_reward']}
└─ Confidence: {setup['confidence']}
"""
    else:
        msg += "\n⏸️ No clear setup available. Wait for better alignment.\n"
    
    msg += f"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
⚠️ **RISK ASSESSMENT**
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Risk Level: {result['risk_rating']['rating']}
Risk Score: {result['risk_rating']['score']}/{result['risk_rating']['max_score']}
Confidence: {result['risk_rating']['confidence']}

╚══════════════════════════════════════╝
"""
    
    return msg


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 6: EXAMPLE USAGE
# ═══════════════════════════════════════════════════════════════════════════════

def example_usage():
    """Example of how to use the library with JSON input"""
    
    # Sample JSON input (this is what your bot will send)
    sample_json = {
        "symbol": "NIFTY",
        "current_price": 44520.0,
        "expiry_date": "2025-11-28",
        
        "candles": {
            "5min": [
                {
                    "timestamp": "2025-11-09 09:15:00",
                    "open": 44480.0,
                    "high": 44530.0,
                    "low": 44460.0,
                    "close": 44500.0,
                    "volume": 150000
                },
                {
                    "timestamp": "2025-11-09 09:20:00",
                    "open": 44500.0,
                    "high": 44550.0,
                    "low": 44490.0,
                    "close": 44520.0,
                    "volume": 180000
                },
                # Add more candles... (200 total for best results)
            ],
            
            "15min": [
                {
                    "timestamp": "2025-11-09 09:00:00",
                    "open": 44450.0,
                    "high": 44550.0,
                    "low": 44430.0,
                    "close": 44520.0,
                    "volume": 450000
                },
                # Add more candles... (175 total for best results)
            ],
            
            "1hour": [
                {
                    "timestamp": "2025-11-09 09:00:00",
                    "open": 44400.0,
                    "high": 44600.0,
                    "low": 44350.0,
                    "close": 44520.0,
                    "volume": 1800000
                },
                # Add more candles... (50 total minimum)
            ]
        },
        
        "oi_data": {
            "current": {
                "timestamp": "2025-11-09 10:30:00",
                "strikes": [
                    {"strike": 44000, "ce_oi": 1500000, "pe_oi": 800000, "ce_volume": 50000, "pe_volume": 30000},
                    {"strike": 44100, "ce_oi": 1400000, "pe_oi": 900000, "ce_volume": 45000, "pe_volume": 35000},
                    {"strike": 44200, "ce_oi": 1600000, "pe_oi": 1000000, "ce_volume": 55000, "pe_volume": 40000},
                    {"strike": 44300, "ce_oi": 1800000, "pe_oi": 1100000, "ce_volume": 60000, "pe_volume": 45000},
                    {"strike": 44400, "ce_oi": 2000000, "pe_oi": 1200000, "ce_volume": 70000, "pe_volume": 50000},
                    {"strike": 44500, "ce_oi": 2500000, "pe_oi": 2800000, "ce_volume": 100000, "pe_volume": 120000},
                    {"strike": 44600, "ce_oi": 1200000, "pe_oi": 2000000, "ce_volume": 50000, "pe_volume": 70000},
                    {"strike": 44700, "ce_oi": 1000000, "pe_oi": 1800000, "ce_volume": 40000, "pe_volume": 60000},
                    {"strike": 44800, "ce_oi": 900000, "pe_oi": 1600000, "ce_volume": 35000, "pe_volume": 55000},
                    {"strike": 44900, "ce_oi": 800000, "pe_oi": 1400000, "ce_volume": 30000, "pe_volume": 45000},
                ]
            },
            
            "fifteen_min_ago": {
                "timestamp": "2025-11-09 10:15:00",
                "strikes": [
                    {"strike": 44000, "ce_oi": 1450000, "pe_oi": 750000},
                    {"strike": 44100, "ce_oi": 1350000, "pe_oi": 850000},
                    {"strike": 44200, "ce_oi": 1550000, "pe_oi": 950000},
                    {"strike": 44300, "ce_oi": 1750000, "pe_oi": 1050000},
                    {"strike": 44400, "ce_oi": 1950000, "pe_oi": 1150000},
                    {"strike": 44500, "ce_oi": 2400000, "pe_oi": 2700000},
                    {"strike": 44600, "ce_oi": 1150000, "pe_oi": 1950000},
                    {"strike": 44700, "ce_oi": 950000, "pe_oi": 1750000},
                    {"strike": 44800, "ce_oi": 850000, "pe_oi": 1550000},
                    {"strike": 44900, "ce_oi": 750000, "pe_oi": 1350000},
                ]
            },
            
            "thirty_min_ago": {
                "timestamp": "2025-11-09 10:00:00",
                "strikes": [
                    {"strike": 44000, "ce_oi": 1400000, "pe_oi": 700000},
                    {"strike": 44100, "ce_oi": 1300000, "pe_oi": 800000},
                    {"strike": 44200, "ce_oi": 1500000, "pe_oi": 900000},
                    {"strike": 44300, "ce_oi": 1700000, "pe_oi": 1000000},
                    {"strike": 44400, "ce_oi": 1900000, "pe_oi": 1100000},
                    {"strike": 44500, "ce_oi": 2300000, "pe_oi": 2600000},
                    {"strike": 44600, "ce_oi": 1100000, "pe_oi": 1900000},
                    {"strike": 44700, "ce_oi": 900000, "pe_oi": 1700000},
                    {"strike": 44800, "ce_oi": 800000, "pe_oi": 1500000},
                    {"strike": 44900, "ce_oi": 700000, "pe_oi": 1300000},
                ]
            }
        }
    }
    
    # Initialize analyzer
    analyzer = MarketFusionAnalyzer()
    
    # Run analysis
    print("Running analysis...")
    result = analyzer.analyze_from_json(sample_json)
    
    # Print formatted output
    print(format_output_for_bot(result))
    
    # Also print raw JSON result if needed
    print("\n" + "="*80)
    print("RAW JSON OUTPUT (for debugging):")
    print("="*80)
    print(json.dumps(result, indent=2))


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 7: MAIN ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("""
    ╔══════════════════════════════════════════════════════════════╗
    ║                                                              ║
    ║        🎯 MARKET FUSION ANALYZER v1.0 🎯                    ║
    ║                                                              ║
    ║   Price Action + OI Intelligence Combined Analysis          ║
    ║                                                              ║
    ╚══════════════════════════════════════════════════════════════╝
    """)
    
    # Run example
    example_usage()
    
    print("\n✅ Analysis complete!")
    print("\n📝 Usage in your bot:")
    print("""
    from main import MarketFusionAnalyzer, format_output_for_bot
    
    analyzer = MarketFusionAnalyzer()
    result = analyzer.analyze_from_json(your_json_data)
    bot_message = format_output_for_bot(result)
    
    # Send to Telegram/Discord
    send_message(bot_message)
    """)
