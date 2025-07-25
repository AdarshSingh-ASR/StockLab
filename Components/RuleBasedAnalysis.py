"""
Rule-Based Analysis System
==========================

This module provides:
1. Rule-based signal generation for each investment strategy
2. Statistical analysis and scoring systems
3. Pre-computed analysis templates
4. Custom decision trees for investment decisions
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple
from dataclasses import dataclass
import logging

@dataclass
class InvestmentSignal:
    """Structured output for investment decisions"""
    signal: str  # "bullish", "bearish", "neutral"
    confidence: float  # 0.0 to 1.0
    reasoning: str  # Human-readable explanation
    key_factors: List[str]  # List of key factors that influenced decision

class RuleBasedAnalyzer:
    """Base class for rule-based investment analysis"""
    
    def __init__(self, agent_name: str):
        self.agent_name = agent_name
        self.logger = logging.getLogger(f"{agent_name}Analyzer")
    
    def analyze_fundamentals(self, financial_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze fundamental metrics and return scores"""
        raise NotImplementedError
    
    def generate_signal(self, analysis_data: Dict[str, Any]) -> InvestmentSignal:
        """Generate investment signal based on analysis"""
        raise NotImplementedError

class WarrenBuffettAnalyzer(RuleBasedAnalyzer):
    """Rule-based analysis following Warren Buffett's principles"""
    
    def __init__(self):
        super().__init__("Warren Buffett")
    
    def analyze_fundamentals(self, financial_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze fundamentals using Buffett's criteria"""
        score = 0
        factors = []
        
        # ROE Analysis (25% weight)
        if 'return_on_equity' in financial_data:
            roe = financial_data['return_on_equity']
            if roe > 0.15:  # 15%+ ROE is excellent
                score += 2.5
                factors.append(f"Strong ROE: {roe:.1%}")
            elif roe > 0.10:  # 10-15% ROE is good
                score += 1.5
                factors.append(f"Good ROE: {roe:.1%}")
            elif roe > 0.05:  # 5-10% ROE is acceptable
                score += 0.5
                factors.append(f"Acceptable ROE: {roe:.1%}")
        
        # Debt Analysis (20% weight)
        if 'debt_to_equity' in financial_data:
            debt_equity = financial_data['debt_to_equity']
            if debt_equity < 0.3:  # Low debt is preferred
                score += 2.0
                factors.append(f"Low debt ratio: {debt_equity:.2f}")
            elif debt_equity < 0.5:  # Moderate debt
                score += 1.0
                factors.append(f"Moderate debt ratio: {debt_equity:.2f}")
            elif debt_equity > 1.0:  # High debt
                score -= 1.0
                factors.append(f"High debt ratio: {debt_equity:.2f}")
        
        # Operating Margin (20% weight)
        if 'operating_margin' in financial_data:
            op_margin = financial_data['operating_margin']
            if op_margin > 0.20:  # 20%+ margin is excellent
                score += 2.0
                factors.append(f"Excellent operating margin: {op_margin:.1%}")
            elif op_margin > 0.15:  # 15-20% margin is good
                score += 1.5
                factors.append(f"Good operating margin: {op_margin:.1%}")
            elif op_margin > 0.10:  # 10-15% margin is acceptable
                score += 1.0
                factors.append(f"Acceptable operating margin: {op_margin:.1%}")
        
        # Current Ratio (15% weight)
        if 'current_ratio' in financial_data:
            curr_ratio = financial_data['current_ratio']
            if curr_ratio > 1.5:  # Strong liquidity
                score += 1.5
                factors.append(f"Strong liquidity: {curr_ratio:.2f}")
            elif curr_ratio > 1.0:  # Adequate liquidity
                score += 0.75
                factors.append(f"Adequate liquidity: {curr_ratio:.2f}")
            elif curr_ratio < 0.8:  # Poor liquidity
                score -= 1.0
                factors.append(f"Poor liquidity: {curr_ratio:.2f}")
        
        # Consistency Analysis (20% weight)
        consistency_score = self._analyze_consistency(financial_data)
        score += consistency_score
        if consistency_score > 1.0:
            factors.append("Strong earnings consistency")
        elif consistency_score < 0:
            factors.append("Earnings volatility concerns")
        
        return {
            "score": score,
            "max_score": 10.0,
            "factors": factors,
            "details": f"Fundamental analysis score: {score:.1f}/10.0"
        }
    
    def _analyze_consistency(self, financial_data: Dict[str, Any]) -> float:
        """Analyze earnings consistency"""
        score = 0
        
        # Check for consistent positive earnings
        if 'net_income' in financial_data:
            net_income = financial_data['net_income']
            if isinstance(net_income, list) and len(net_income) >= 3:
                positive_years = sum(1 for ni in net_income if ni > 0)
                if positive_years == len(net_income):
                    score += 1.0  # All years positive
                elif positive_years >= len(net_income) * 0.8:
                    score += 0.5  # Mostly positive
        
        return score
    
    def generate_signal(self, analysis_data: Dict[str, Any]) -> InvestmentSignal:
        """Generate Buffett-style investment signal"""
        total_score = analysis_data.get("score", 0)
        max_score = analysis_data.get("max_score", 10.0)
        margin_of_safety = analysis_data.get("margin_of_safety", 0)
        
        # Calculate confidence based on score strength
        confidence = min(total_score / max_score, 1.0)
        
        # Generate signal based on Buffett's criteria
        if total_score >= 0.7 * max_score and margin_of_safety >= 0.3:
            signal = "bullish"
            reasoning = f"Strong fundamentals with {total_score:.1f}/10.0 score and {margin_of_safety:.1%} margin of safety. Meets Buffett's criteria for value investing."
        elif total_score <= 0.3 * max_score or (margin_of_safety is not None and margin_of_safety < -0.3):
            signal = "bearish"
            reasoning = f"Weak fundamentals with {total_score:.1f}/10.0 score. Does not meet Buffett's investment criteria."
        else:
            signal = "neutral"
            reasoning = f"Mixed signals with {total_score:.1f}/10.0 score. Requires further analysis."
        
        return InvestmentSignal(
            signal=signal,
            confidence=confidence,
            reasoning=reasoning,
            key_factors=analysis_data.get("factors", [])
        )

class PeterLynchAnalyzer(RuleBasedAnalyzer):
    """Rule-based analysis following Peter Lynch's principles"""
    
    def __init__(self):
        super().__init__("Peter Lynch")
    
    def analyze_fundamentals(self, financial_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze fundamentals using Lynch's GARP approach"""
        score = 0
        factors = []
        
        # PEG Ratio Analysis (30% weight)
        if 'peg_ratio' in financial_data:
            peg = financial_data['peg_ratio']
            if peg < 1.0:  # Undervalued growth
                score += 3.0
                factors.append(f"Undervalued growth (PEG: {peg:.2f})")
            elif peg < 1.5:  # Reasonable growth
                score += 2.0
                factors.append(f"Reasonable growth (PEG: {peg:.2f})")
            elif peg > 2.0:  # Overvalued
                score -= 1.0
                factors.append(f"Overvalued growth (PEG: {peg:.2f})")
        
        # Revenue Growth (25% weight)
        if 'revenue_growth' in financial_data:
            growth = financial_data['revenue_growth']
            if growth > 0.20:  # 20%+ growth
                score += 2.5
                factors.append(f"Strong revenue growth: {growth:.1%}")
            elif growth > 0.10:  # 10-20% growth
                score += 1.5
                factors.append(f"Good revenue growth: {growth:.1%}")
            elif growth < 0.05:  # Low growth
                score -= 1.0
                factors.append(f"Low revenue growth: {growth:.1%}")
        
        # Earnings Growth (25% weight)
        if 'earnings_growth' in financial_data:
            earnings_growth = financial_data['earnings_growth']
            if earnings_growth > 0.15:  # 15%+ earnings growth
                score += 2.5
                factors.append(f"Strong earnings growth: {earnings_growth:.1%}")
            elif earnings_growth > 0.08:  # 8-15% earnings growth
                score += 1.5
                factors.append(f"Good earnings growth: {earnings_growth:.1%}")
            elif earnings_growth < 0.05:  # Low earnings growth
                score -= 1.0
                factors.append(f"Low earnings growth: {earnings_growth:.1%}")
        
        # Market Cap Analysis (20% weight)
        if 'market_cap' in financial_data:
            market_cap = financial_data['market_cap']
            if market_cap < 1e9:  # Small cap potential
                score += 2.0
                factors.append("Small cap growth potential")
            elif market_cap < 1e10:  # Mid cap
                score += 1.0
                factors.append("Mid cap company")
            else:  # Large cap
                score += 0.5
                factors.append("Large cap company")
        
        return {
            "score": score,
            "max_score": 10.0,
            "factors": factors,
            "details": f"GARP analysis score: {score:.1f}/10.0"
        }
    
    def generate_signal(self, analysis_data: Dict[str, Any]) -> InvestmentSignal:
        """Generate Lynch-style investment signal"""
        total_score = analysis_data.get("score", 0)
        max_score = analysis_data.get("max_score", 10.0)
        
        confidence = min(total_score / max_score, 1.0)
        
        if total_score >= 0.7 * max_score:
            signal = "bullish"
            reasoning = f"Strong growth at reasonable price with {total_score:.1f}/10.0 score. Meets Lynch's GARP criteria."
        elif total_score <= 0.3 * max_score:
            signal = "bearish"
            reasoning = f"Weak growth fundamentals with {total_score:.1f}/10.0 score. Does not meet Lynch's criteria."
        else:
            signal = "neutral"
            reasoning = f"Mixed growth signals with {total_score:.1f}/10.0 score. Requires further analysis."
        
        return InvestmentSignal(
            signal=signal,
            confidence=confidence,
            reasoning=reasoning,
            key_factors=analysis_data.get("factors", [])
        )

class CathieWoodAnalyzer(RuleBasedAnalyzer):
    """Rule-based analysis following Cathie Wood's principles"""
    
    def __init__(self):
        super().__init__("Cathie Wood")
    
    def analyze_fundamentals(self, financial_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze fundamentals using Wood's innovation focus"""
        score = 0
        factors = []
        
        # R&D Investment (30% weight)
        if 'rd_to_revenue' in financial_data:
            rd_ratio = financial_data['rd_to_revenue']
            if rd_ratio > 0.15:  # High R&D investment
                score += 3.0
                factors.append(f"High R&D investment: {rd_ratio:.1%}")
            elif rd_ratio > 0.08:  # Moderate R&D
                score += 1.5
                factors.append(f"Moderate R&D investment: {rd_ratio:.1%}")
            elif rd_ratio < 0.03:  # Low R&D
                score -= 1.0
                factors.append(f"Low R&D investment: {rd_ratio:.1%}")
        
        # Revenue Growth (25% weight)
        if 'revenue_growth' in financial_data:
            growth = financial_data['revenue_growth']
            if growth > 0.30:  # 30%+ growth
                score += 2.5
                factors.append(f"Exceptional revenue growth: {growth:.1%}")
            elif growth > 0.20:  # 20-30% growth
                score += 2.0
                factors.append(f"Strong revenue growth: {growth:.1%}")
            elif growth > 0.10:  # 10-20% growth
                score += 1.0
                factors.append(f"Good revenue growth: {growth:.1%}")
        
        # Market Disruption Potential (25% weight)
        if 'market_cap' in financial_data and 'total_addressable_market' in financial_data:
            market_cap = financial_data['market_cap']
            tam = financial_data['total_addressable_market']
            if tam > 0 and market_cap / tam < 0.01:  # Small market share
                score += 2.5
                factors.append("Large TAM opportunity")
            elif market_cap / tam < 0.05:  # Moderate market share
                score += 1.5
                factors.append("Significant TAM opportunity")
        
        # Innovation Metrics (20% weight)
        if 'patents' in financial_data:
            patents = financial_data['patents']
            if patents > 100:  # High patent count
                score += 2.0
                factors.append(f"Strong IP portfolio: {patents} patents")
            elif patents > 50:  # Moderate patent count
                score += 1.0
                factors.append(f"Good IP portfolio: {patents} patents")
        
        return {
            "score": score,
            "max_score": 10.0,
            "factors": factors,
            "details": f"Innovation analysis score: {score:.1f}/10.0"
        }
    
    def generate_signal(self, analysis_data: Dict[str, Any]) -> InvestmentSignal:
        """Generate Wood-style investment signal"""
        total_score = analysis_data.get("score", 0)
        max_score = analysis_data.get("max_score", 10.0)
        
        confidence = min(total_score / max_score, 1.0)
        
        if total_score >= 0.7 * max_score:
            signal = "bullish"
            reasoning = f"Strong innovation potential with {total_score:.1f}/10.0 score. Meets Wood's disruptive innovation criteria."
        elif total_score <= 0.3 * max_score:
            signal = "bearish"
            reasoning = f"Limited innovation potential with {total_score:.1f}/10.0 score. Does not meet Wood's criteria."
        else:
            signal = "neutral"
            reasoning = f"Mixed innovation signals with {total_score:.1f}/10.0 score. Requires further analysis."
        
        return InvestmentSignal(
            signal=signal,
            confidence=confidence,
            reasoning=reasoning,
            key_factors=analysis_data.get("factors", [])
        )

class BenGrahamAnalyzer(RuleBasedAnalyzer):
    """Rule-based analysis following Ben Graham's principles"""
    
    def __init__(self):
        super().__init__("Ben Graham")
    
    def analyze_fundamentals(self, financial_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze fundamentals using Graham's value approach"""
        score = 0
        factors = []
        
        # P/B Ratio Analysis (30% weight)
        if 'price_to_book' in financial_data:
            pb_ratio = financial_data['price_to_book']
            if pb_ratio < 1.0:  # Trading below book value
                score += 3.0
                factors.append(f"Trading below book value (P/B: {pb_ratio:.2f})")
            elif pb_ratio < 1.5:  # Reasonable P/B
                score += 2.0
                factors.append(f"Reasonable P/B ratio: {pb_ratio:.2f}")
            elif pb_ratio > 3.0:  # Expensive
                score -= 1.0
                factors.append(f"Expensive P/B ratio: {pb_ratio:.2f}")
        
        # Net Current Asset Value (25% weight)
        if 'net_current_assets' in financial_data and 'market_cap' in financial_data:
            nca = financial_data['net_current_assets']
            market_cap = financial_data['market_cap']
            if nca > market_cap:  # NCA > Market Cap
                score += 2.5
                factors.append("Net current assets exceed market cap")
            elif nca > 0.8 * market_cap:  # NCA close to Market Cap
                score += 1.5
                factors.append("Strong net current asset position")
        
        # Debt Analysis (25% weight)
        if 'debt_to_equity' in financial_data:
            debt_equity = financial_data['debt_to_equity']
            if debt_equity < 0.2:  # Low debt
                score += 2.5
                factors.append(f"Low debt ratio: {debt_equity:.2f}")
            elif debt_equity < 0.5:  # Moderate debt
                score += 1.0
                factors.append(f"Moderate debt ratio: {debt_equity:.2f}")
            elif debt_equity > 1.0:  # High debt
                score -= 1.0
                factors.append(f"High debt ratio: {debt_equity:.2f}")
        
        # Earnings Stability (20% weight)
        if 'earnings_stability' in financial_data:
            stability = financial_data['earnings_stability']
            if stability > 0.8:  # High stability
                score += 2.0
                factors.append("High earnings stability")
            elif stability > 0.6:  # Moderate stability
                score += 1.0
                factors.append("Moderate earnings stability")
        
        return {
            "score": score,
            "max_score": 10.0,
            "factors": factors,
            "details": f"Value analysis score: {score:.1f}/10.0"
        }
    
    def generate_signal(self, analysis_data: Dict[str, Any]) -> InvestmentSignal:
        """Generate Graham-style investment signal"""
        total_score = analysis_data.get("score", 0)
        max_score = analysis_data.get("max_score", 10.0)
        
        confidence = min(total_score / max_score, 1.0)
        
        if total_score >= 0.7 * max_score:
            signal = "bullish"
            reasoning = f"Strong value characteristics with {total_score:.1f}/10.0 score. Meets Graham's value investing criteria."
        elif total_score <= 0.3 * max_score:
            signal = "bearish"
            reasoning = f"Weak value characteristics with {total_score:.1f}/10.0 score. Does not meet Graham's criteria."
        else:
            signal = "neutral"
            reasoning = f"Mixed value signals with {total_score:.1f}/10.0 score. Requires further analysis."
        
        return InvestmentSignal(
            signal=signal,
            confidence=confidence,
            reasoning=reasoning,
            key_factors=analysis_data.get("factors", [])
        )

class StatisticalReportGenerator:
    """Generates statistical analysis reports without LLM dependencies"""
    
    @staticmethod
    def generate_performance_report(summary_data: Dict[str, Any]) -> str:
        """Generate performance analysis report"""
        report = "# Performance Analysis Report\n\n"
        
        # Model comparison
        if 'model_results' in summary_data:
            report += "## Model Performance Comparison\n\n"
            for model_name, results in summary_data['model_results'].items():
                sharpe = results.get('sharpe_ratio', 0)
                max_dd = results.get('max_drawdown', 0)
                total_return = results.get('total_return', 0)
                
                report += f"### {model_name}\n"
                report += f"- **Sharpe Ratio**: {sharpe:.3f}\n"
                report += f"- **Max Drawdown**: {max_dd:.1%}\n"
                report += f"- **Total Return**: {total_return:.1%}\n\n"
        
        # Risk analysis
        if 'risk_metrics' in summary_data:
            report += "## Risk Analysis\n\n"
            risk_metrics = summary_data['risk_metrics']
            report += f"- **Volatility**: {risk_metrics.get('volatility', 0):.1%}\n"
            report += f"- **VaR (95%)**: {risk_metrics.get('var_95', 0):.1%}\n"
            report += f"- **Beta**: {risk_metrics.get('beta', 0):.3f}\n\n"
        
        return report
    
    @staticmethod
    def generate_recommendations(summary_data: Dict[str, Any]) -> str:
        """Generate investment recommendations"""
        report = "# Investment Recommendations\n\n"
        
        # Find best performing model
        best_model = None
        best_sharpe = -999
        
        if 'model_results' in summary_data:
            for model_name, results in summary_data['model_results'].items():
                sharpe = results.get('sharpe_ratio', 0)
                if sharpe > best_sharpe:
                    best_sharpe = sharpe
                    best_model = model_name
        
        if best_model:
            report += f"## Primary Recommendation\n\n"
            report += f"**{best_model}** shows the strongest risk-adjusted returns with a Sharpe ratio of {best_sharpe:.3f}.\n\n"
            
            if best_sharpe > 1.0:
                report += "✅ **Strong Buy**: This model demonstrates excellent risk-adjusted performance.\n"
            elif best_sharpe > 0.5:
                report += "✅ **Buy**: This model shows good risk-adjusted performance.\n"
            else:
                report += "⚠️ **Hold**: This model shows moderate performance.\n"
        
        # Risk considerations
        report += "\n## Risk Considerations\n\n"
        report += "- Monitor market regime changes\n"
        report += "- Consider transaction costs in live trading\n"
        report += "- Diversify across multiple strategies\n"
        report += "- Regular rebalancing recommended\n"
        
        return report

# Simple analyzer classes for missing agents
class CharlieMungerAnalyzer(RuleBasedAnalyzer):
    """Rule-based analysis following Charlie Munger's principles"""
    
    def __init__(self):
        super().__init__("Charlie Munger")
    
    def generate_signal(self, analysis_data: Dict[str, Any]) -> InvestmentSignal:
        score = analysis_data.get("score", 0)
        max_score = analysis_data.get("max_score", 10)
        
        if score >= 0.7 * max_score:
            signal = "bullish"
            reasoning = f"Strong mental model analysis with {score:.1f}/10.0 score. Meets Munger's criteria."
        elif score <= 0.3 * max_score:
            signal = "bearish"
            reasoning = f"Weak mental model analysis with {score:.1f}/10.0 score. Does not meet Munger's criteria."
        else:
            signal = "neutral"
            reasoning = f"Mixed mental model signals with {score:.1f}/10.0 score. Requires further analysis."
        
        return InvestmentSignal(
            signal=signal,
            confidence=0.8,
            reasoning=reasoning,
            key_factors=analysis_data.get("factors", [])
        )

class BillAckmanAnalyzer(RuleBasedAnalyzer):
    """Rule-based analysis following Bill Ackman's principles"""
    
    def __init__(self):
        super().__init__("Bill Ackman")
    
    def generate_signal(self, analysis_data: Dict[str, Any]) -> InvestmentSignal:
        score = analysis_data.get("score", 0)
        max_score = analysis_data.get("max_score", 10)
        
        if score >= 0.7 * max_score:
            signal = "bullish"
            reasoning = f"Strong concentrated position analysis with {score:.1f}/10.0 score. Meets Ackman's criteria."
        elif score <= 0.3 * max_score:
            signal = "bearish"
            reasoning = f"Weak concentrated position analysis with {score:.1f}/10.0 score. Does not meet Ackman's criteria."
        else:
            signal = "neutral"
            reasoning = f"Mixed concentrated position signals with {score:.1f}/10.0 score. Requires further analysis."
        
        return InvestmentSignal(
            signal=signal,
            confidence=0.8,
            reasoning=reasoning,
            key_factors=analysis_data.get("factors", [])
        )

class StanleyDruckenmillerAnalyzer(RuleBasedAnalyzer):
    """Rule-based analysis following Stanley Druckenmiller's principles"""
    
    def __init__(self):
        super().__init__("Stanley Druckenmiller")
    
    def generate_signal(self, analysis_data: Dict[str, Any]) -> InvestmentSignal:
        score = analysis_data.get("score", 0)
        max_score = analysis_data.get("max_score", 10)
        
        if score >= 0.7 * max_score:
            signal = "bullish"
            reasoning = f"Strong macro trend analysis with {score:.1f}/10.0 score. Meets Druckenmiller's criteria."
        elif score <= 0.3 * max_score:
            signal = "bearish"
            reasoning = f"Weak macro trend analysis with {score:.1f}/10.0 score. Does not meet Druckenmiller's criteria."
        else:
            signal = "neutral"
            reasoning = f"Mixed macro trend signals with {score:.1f}/10.0 score. Requires further analysis."
        
        return InvestmentSignal(
            signal=signal,
            confidence=0.8,
            reasoning=reasoning,
            key_factors=analysis_data.get("factors", [])
        )

class PhilFisherAnalyzer(RuleBasedAnalyzer):
    """Rule-based analysis following Phil Fisher's principles"""
    
    def __init__(self):
        super().__init__("Phil Fisher")
    
    def generate_signal(self, analysis_data: Dict[str, Any]) -> InvestmentSignal:
        score = analysis_data.get("score", 0)
        max_score = analysis_data.get("max_score", 10)
        
        if score >= 0.7 * max_score:
            signal = "bullish"
            reasoning = f"Strong qualitative analysis with {score:.1f}/10.0 score. Meets Fisher's criteria."
        elif score <= 0.3 * max_score:
            signal = "bearish"
            reasoning = f"Weak qualitative analysis with {score:.1f}/10.0 score. Does not meet Fisher's criteria."
        else:
            signal = "neutral"
            reasoning = f"Mixed qualitative signals with {score:.1f}/10.0 score. Requires further analysis."
        
        return InvestmentSignal(
            signal=signal,
            confidence=0.8,
            reasoning=reasoning,
            key_factors=analysis_data.get("factors", [])
        )

class AswathDamodaranAnalyzer(RuleBasedAnalyzer):
    """Rule-based analysis following Aswath Damodaran's principles"""
    
    def __init__(self):
        super().__init__("Aswath Damodaran")
    
    def generate_signal(self, analysis_data: Dict[str, Any]) -> InvestmentSignal:
        score = analysis_data.get("score", 0)
        max_score = analysis_data.get("max_score", 10)
        
        if score >= 0.7 * max_score:
            signal = "bullish"
            reasoning = f"Strong valuation analysis with {score:.1f}/10.0 score. Meets Damodaran's criteria."
        elif score <= 0.3 * max_score:
            signal = "bearish"
            reasoning = f"Weak valuation analysis with {score:.1f}/10.0 score. Does not meet Damodaran's criteria."
        else:
            signal = "neutral"
            reasoning = f"Mixed valuation signals with {score:.1f}/10.0 score. Requires further analysis."
        
        return InvestmentSignal(
            signal=signal,
            confidence=0.8,
            reasoning=reasoning,
            key_factors=analysis_data.get("factors", [])
        )

class ValuationAnalyzer(RuleBasedAnalyzer):
    """Rule-based analysis for valuation agent"""
    
    def __init__(self):
        super().__init__("Valuation")
    
    def generate_signal(self, analysis_data: Dict[str, Any]) -> InvestmentSignal:
        score = analysis_data.get("score", 0)
        max_score = analysis_data.get("max_score", 10)
        
        if score >= 0.7 * max_score:
            signal = "bullish"
            reasoning = f"Strong multi-factor valuation with {score:.1f}/10.0 score. Attractive valuation metrics."
        elif score <= 0.3 * max_score:
            signal = "bearish"
            reasoning = f"Weak multi-factor valuation with {score:.1f}/10.0 score. Unattractive valuation metrics."
        else:
            signal = "neutral"
            reasoning = f"Mixed valuation signals with {score:.1f}/10.0 score. Requires further analysis."
        
        return InvestmentSignal(
            signal=signal,
            confidence=0.8,
            reasoning=reasoning,
            key_factors=analysis_data.get("factors", [])
        )

class FundamentalsAnalyzer(RuleBasedAnalyzer):
    """Rule-based analysis for fundamentals agent"""
    
    def __init__(self):
        super().__init__("Fundamentals")
    
    def generate_signal(self, analysis_data: Dict[str, Any]) -> InvestmentSignal:
        score = analysis_data.get("score", 0)
        max_score = analysis_data.get("max_score", 10)
        
        if score >= 0.7 * max_score:
            signal = "bullish"
            reasoning = f"Strong financial metrics with {score:.1f}/10.0 score. Solid fundamental indicators."
        elif score <= 0.3 * max_score:
            signal = "bearish"
            reasoning = f"Weak financial metrics with {score:.1f}/10.0 score. Poor fundamental indicators."
        else:
            signal = "neutral"
            reasoning = f"Mixed financial metrics with {score:.1f}/10.0 score. Requires further analysis."
        
        return InvestmentSignal(
            signal=signal,
            confidence=0.8,
            reasoning=reasoning,
            key_factors=analysis_data.get("factors", [])
        )

# Factory function to get the appropriate analyzer
def get_analyzer(agent_name: str) -> RuleBasedAnalyzer:
    """Get the appropriate rule-based analyzer for an agent"""
    analyzers = {
        "WarrenBuffettAgent": WarrenBuffettAnalyzer,
        "PeterLynchAgent": PeterLynchAnalyzer,
        "CathieWoodAgent": CathieWoodAnalyzer,
        "BenGrahamAgent": BenGrahamAnalyzer,
        "CharlieMungerAgent": CharlieMungerAnalyzer,
        "BillAckmanAgent": BillAckmanAnalyzer,
        "StanleyDruckenmillerAgent": StanleyDruckenmillerAnalyzer,
        "PhilFisherAgent": PhilFisherAnalyzer,
        "AswathDamodaranAgent": AswathDamodaranAnalyzer,
        "ValuationAgent": ValuationAnalyzer,
        "FundamentalsAgent": FundamentalsAnalyzer,
    }
    
    analyzer_class = analyzers.get(agent_name)
    if analyzer_class:
        return analyzer_class()
    else:
        # Default analyzer for unknown agents
        return RuleBasedAnalyzer(agent_name) 