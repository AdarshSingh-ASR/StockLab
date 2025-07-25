import math
from pandas import DataFrame
import pandas as pd
import numpy as np
import logging
import time
import os

from Components.Fundamentals import search_line_items, get_metric_value
from Components.RuleBasedAnalysis import get_analyzer

class WarrenBuffettAgent:
    """Analyzes stocks using Buffett's principles and LLM reasoning."""
    def __init__(self, ticker, metrics, **kwargs):
        self.agent_name = "Warren Buffett"
        self.metrics = metrics
        self.ticker = ticker
        self.period = kwargs.get('analysis_period')
        self.limit = kwargs.get('analysis_limit')
        self.threshold_matrix_path = kwargs.get('threshold_matrix_path',None)
        self.model_name = kwargs.get('model_name','rule-based-analyzer')
        self.analysis_data = {} # Storing returned results in dict  
        
    def analyze(self):
        # Optimized for speed - use only essential metrics
        financial_line_items, self.SIC_code = search_line_items(
            self.ticker,
            [
                "return_on_equity",
                "debt_to_equity", 
                "operating_margin",
                "current_ratio",
                "net_income",
                "market_cap",
            ],
            period=self.period,
            limit=min(4, self.limit),  # Reduce historical data for speed
            df=self.metrics
        )
        
        self.threshold_matrix = pd.read_csv(self.threshold_matrix_path.get('business_services_sic')) if len(self.SIC_code) > 2 else pd.read_csv(self.threshold_matrix_path.get('two_digit_sic'))
        
        fundamental_analysis = self.analyze_fundamentals(financial_line_items)
        consistency_analysis = self.analyze_consistency(financial_line_items)
        moat_analysis = self.analyze_moat(financial_line_items)
        mgmt_analysis = self.analyze_management_quality(financial_line_items)
        # Skip intrinsic value calculation for speed
        intrinsic_value_analysis = {"intrinsic_value": None, "margin_of_safety": None}

        # Calculate total score
        total_score = fundamental_analysis["score"] + consistency_analysis["score"] + moat_analysis["score"] + mgmt_analysis["score"]
        max_possible_score = 10 + moat_analysis["max_score"] + mgmt_analysis["max_score"]
        
        # Cap the score at 10 for display purposes
        display_score = min(total_score, 10.0)
        # fundamental_analysis + consistency combined were up to 10 points total
        # moat can add up to 3, mgmt can add up to 2, for example
        intrinsic_value = intrinsic_value_analysis["intrinsic_value"]
        margin_of_safety = intrinsic_value_analysis["margin_of_safety"]
        
        # Generate trading signal using a stricter margin-of-safety requirement
        # if fundamentals+moat+management are strong but margin_of_safety < 0.3, it's neutral
        # if fundamentals are very weak or margin_of_safety is severely negative -> bearish
        # else bullish
        if (total_score >= 0.7 * max_possible_score) and margin_of_safety and (margin_of_safety >= 0.3):
            signal = "bullish"
        elif total_score <= 0.3 * max_possible_score or (margin_of_safety is not None and margin_of_safety < -0.3):
            # negative margin of safety beyond -30% could be overpriced -> bearish
            signal = "bearish"
        else:
            signal = "neutral"

        # Combine all analysis results
        self.analysis_data = {
            "name": self.agent_name,
            #"signal": signal,
            "score": display_score,
            "max_score": 10.0,
            "fundamental_analysis": fundamental_analysis,
            "consistency_analysis": consistency_analysis,
            "moat_analysis": moat_analysis,
            "management_analysis": mgmt_analysis,
            "intrinsic_value_analysis": intrinsic_value_analysis,
            "margin_of_safety": margin_of_safety,
        }

        # Use rule-based analysis instead of LLM
        analyzer = get_analyzer("WarrenBuffettAgent")
        rule_based_signal = analyzer.generate_signal(self.analysis_data)

        return {"name":self.analysis_data["name"],"signal": rule_based_signal.signal, "score":self.analysis_data["score"], "confidence": rule_based_signal.confidence, "reasoning": rule_based_signal.reasoning}

    def analyze_fundamentals(self, financial_line_items: DataFrame):
        """Analyze company fundamentals based on Buffett's criteria."""
        if financial_line_items.empty:
            return {"score": 0, "details": "Insufficient fundamental data"}

        score = 0
        reasoning = []

        # Check ROE (Return on Equity)
        roe_threshold = get_metric_value(self.threshold_matrix, self.SIC_code, 'return_on_equity')
        roe_threshold = (roe_threshold*0.5)+roe_threshold
        return_on_equity = financial_line_items.return_on_equity.values[0]
        if return_on_equity is not None and return_on_equity > roe_threshold:  # 15% ROE threshold
            score += 2
            reasoning.append(f"Strong ROE of {return_on_equity:.1%}")
        elif return_on_equity is not None:
            reasoning.append(f"Weak ROE of {return_on_equity:.1%}")
        else:
            reasoning.append("ROE data not available")

        # Check Debt to Equity
        dte_threshold = get_metric_value(self.threshold_matrix, self.SIC_code, 'debt_to_equity')
        debt_to_equity = financial_line_items.debt_to_equity.values[0]
        if debt_to_equity is not None and debt_to_equity < dte_threshold:
            score += 2
            reasoning.append("Conservative debt levels")
        elif debt_to_equity:
            reasoning.append(f"High debt to equity ratio of {debt_to_equity:.1f}")
        else:
            reasoning.append("Debt to equity data not available")

        # Check Operating Margin
        om_threshold = get_metric_value(self.threshold_matrix, self.SIC_code, 'operating_margin')
        om_threshold = (om_threshold*0.5)+om_threshold
        operating_margin = financial_line_items.operating_margin.values[0]
        if operating_margin is not None and operating_margin > om_threshold:
            score += 2
            reasoning.append("Strong operating margins")
        elif operating_margin is not None:
            reasoning.append(f"Weak operating margin of {operating_margin:.1%}")
        else:
            reasoning.append("Operating margin data not available")

        # Check Current Ratio
        current_ratio_threshold = get_metric_value(self.threshold_matrix, self.SIC_code, 'current_ratio')
        current_ratio = financial_line_items.current_ratio.values[0]
        if current_ratio is not None and current_ratio > current_ratio_threshold:
            score += 1
            reasoning.append("Good liquidity position")
        elif current_ratio is not None:
            reasoning.append(f"Weak liquidity with current ratio of {current_ratio:.1f}")
        else:
            reasoning.append("Current ratio data not available")

        return {"score": score, "details": "; ".join(reasoning), "metrics": f"Operating Margin: {operating_margin}; Current Ratio: {current_ratio}; Debt to Equity: {debt_to_equity}; Return on Equity: {return_on_equity}"}


    def analyze_consistency(self, financial_line_items: DataFrame):
        """Analyze earnings consistency and growth."""
        if len(financial_line_items) < 4:  # Need at least 4 periods for trend analysis
            return {"score": 0, "details": "Insufficient historical data"}

        score = 0
        reasoning = []

        # Check earnings growth trend
        earnings_values = financial_line_items.net_income.values
        if len(earnings_values) >= 4:
            # Simple check: is each period's earnings bigger than the next?
            growth_rates = []
            for i in range(len(earnings_values) - 1):
                if earnings_values[i] and earnings_values[i + 1]:
                    growth_rate = (earnings_values[i + 1] - earnings_values[i]) / abs(earnings_values[i]) if earnings_values[i] != 0 else 0
                    growth_rates.append(growth_rate)

            if len(growth_rates) >= 2 and growth_rates[0] > growth_rates[1]:
                score += 3
                reasoning.append("Consistent earnings growth over past periods")
            else:
                reasoning.append("Inconsistent earnings growth pattern")

            # Calculate total growth rate from oldest to latest
            if len(earnings_values) >= 2 and earnings_values[1] != 0:
                growth_rate = (earnings_values[0] - earnings_values[1]) / abs(earnings_values[1])
                reasoning.append(f"Total earnings growth of {growth_rate:.1%} over past {len(earnings_values)} periods")
        else:
            reasoning.append("Insufficient earnings data for trend analysis")

        return {
            "score": score,
            "details": "; ".join(reasoning),
        }

    def analyze_moat(self, financial_line_items: DataFrame):
        """
        Evaluate whether the company likely has a durable competitive advantage (moat).
        For simplicity, we look at stability of ROE/operating margins over multiple periods
        or high margin over the last few years. Higher stability => higher moat score.
        """
        if financial_line_items.empty or len(financial_line_items) < 3:
            return {"score": 0, "max_score": 3, "details": "Insufficient data for moat analysis"}

        reasoning = []
        moat_score = 0

        # Check for stable or improving ROE
        roe_threshold = get_metric_value(self.threshold_matrix, self.SIC_code, 'return_on_equity')
        roe_threshold = (roe_threshold*0.5)+roe_threshold
        historical_roes = financial_line_items.return_on_equity.values
        if len(historical_roes) >= 3:
            stable_roe = all(r > roe_threshold for r in historical_roes)
            if stable_roe:
                moat_score += 1
                reasoning.append(f"Stable ROE above {roe_threshold*100}% across periods (suggests moat)")
            else:
                reasoning.append(f"ROE not consistently above {roe_threshold*100}%")

        # Check for stable or improving operating margin
        om_threshold = get_metric_value(self.threshold_matrix, self.SIC_code, 'operating_margin')
        om_threshold = (om_threshold*0.5)+om_threshold
        historical_margins = financial_line_items.operating_margin.values
        if len(historical_margins) >= 3:
            stable_margin = all(m > om_threshold for m in historical_margins)
            if stable_margin:
                moat_score += 1
                reasoning.append(f"Stable operating margins above {om_threshold*100}% (moat indicator)")
            else:
                reasoning.append(f"Operating margin not consistently above {om_threshold*100}%")

        # If both are stable/improving, add an extra point
        if moat_score == 2:
            moat_score += 1
            reasoning.append("Both ROE and margin stability indicate a solid moat")

        return {
            "score": moat_score,
            "max_score": 3,
            "details": "; ".join(reasoning),
        }

    def analyze_management_quality(self, financial_line_items: DataFrame):
        """
        Checks for share dilution or consistent buybacks, and some dividend track record.
        A simplified approach:
            - if there's net share repurchase or stable share count, it suggests management
            might be shareholder-friendly.
            - if there's a big new issuance, it might be a negative sign (dilution).
        """
        if financial_line_items.empty:
            return {"score": 0, "max_score": 2, "details": "Insufficient data for management analysis"}

        reasoning = []
        mgmt_score = 0

        stock_issuance = financial_line_items.issuance_or_purchase_of_equity_shares.values[0]
        if stock_issuance < 0:
            # Negative means the company spent money on buybacks
            mgmt_score += 1
            reasoning.append("Company has been repurchasing shares (shareholder-friendly)")
        if stock_issuance > 0:
            # Positive issuance means new shares => possible dilution
            reasoning.append("Recent common stock issuance (potential dilution)")
        else:
            reasoning.append("No significant new stock issuance detected")

        # Check for any dividends
        dividends = financial_line_items.dividends_and_other_cash_distributions.values[0]
        if dividends < 0:
            mgmt_score += 1
            reasoning.append("Company has a track record of paying dividends")
        else:
            reasoning.append("No or minimal dividends paid")

        return {
            "score": mgmt_score,
            "max_score": 2,
            "details": "; ".join(reasoning),
        }

    def calculate_owner_earnings(self, financial_line_items: DataFrame):
        """Calculate owner earnings (Buffett's preferred measure of true earnings power).
        Owner Earnings = Net Income + Depreciation - Maintenance CapEx"""
        if financial_line_items.empty or len(financial_line_items) < 1:
            return {"owner_earnings": None, "details": ["Insufficient data for owner earnings calculation"]}

        net_income = financial_line_items.net_income.values[0]
        depreciation = financial_line_items.depreciation_and_amortization.values[0]
        capex = financial_line_items.capital_expenditure.values[0]

        if not net_income and not depreciation and not capex:
            return {"owner_earnings": None, "details": ["Missing components for owner earnings calculation"]}

        # Estimate maintenance capex (typically 70-80% of total capex)
        maintenance_capex = capex * 0.75
        owner_earnings = net_income + depreciation - maintenance_capex

        return {
            "owner_earnings": owner_earnings,
            "components": {"net_income": net_income, "depreciation": depreciation, "maintenance_capex": maintenance_capex},
            "details": ["Owner earnings calculated successfully"],
        }


    def calculate_intrinsic_value(self, financial_line_items: DataFrame):
        """Calculate intrinsic value using DCF with owner earnings."""
        if financial_line_items.empty:
            return {
                "intrinsic_value": None,
                "margin_of_safety": None,
                "details": ["Insufficient data for valuation"]
            }

        # Calculate owner earnings
        earnings_data = self.calculate_owner_earnings(financial_line_items)
        if earnings_data["owner_earnings"] is None:
            return {
                "intrinsic_value": None,
                "margin_of_safety": None,
                "details": earnings_data["details"]
            }

        owner_earnings = earnings_data["owner_earnings"]

        # Get current market data
        shares_outstanding = financial_line_items.outstanding_shares.values[0]
        market_cap = financial_line_items.market_cap.values[0]

        # Buffett's DCF assumptions (conservative approach)
        growth_rate = 0.05  # Conservative 5% growth
        discount_rate = 0.09  # Typical ~9% discount rate
        terminal_multiple = 12
        projection_years = 10

        # Sum of discounted future owner earnings
        future_value = 0
        for year in range(1, projection_years + 1):
            future_earnings = owner_earnings * (1 + growth_rate) ** year
            present_value = future_earnings / (1 + discount_rate) ** year
            future_value += present_value

        # Terminal value
        terminal_value = (owner_earnings * (1 + growth_rate) ** projection_years * terminal_multiple) / ((1 + discount_rate) ** projection_years)

        intrinsic_value = future_value + terminal_value
        if intrinsic_value is not None and market_cap is not None and market_cap != 0:
            margin_of_safety = (intrinsic_value - market_cap) / market_cap
        else:
            margin_of_safety = 0

        return {
            "intrinsic_value": intrinsic_value,
            "margin_of_safety": margin_of_safety,
            "owner_earnings": owner_earnings,
            "assumptions": {
                "growth_rate": growth_rate,
                "discount_rate": discount_rate,
                "terminal_multiple": terminal_multiple,
                "projection_years": projection_years,
            },
            "details": ["Intrinsic value calculated using DCF model with owner earnings"],
        }

    def generate_llm_output(
        self,
        max_retries: int = 3,
        initial_delay: int = 5,
        backoff_factor: float = 1.5
    ):
        """Get investment decision using rule-based analysis (LLM replacement)"""
        try:
            # Use rule-based analysis instead of LLM
            analyzer = get_analyzer("WarrenBuffettAgent")
            
            # Generate signal using rule-based system
            signal = analyzer.generate_signal(self.analysis_data)
            
            return type('obj', (object,), {
                'signal': signal.signal,
                'confidence': signal.confidence,
                'reasoning': signal.reasoning
            })
                
        except Exception as e:
            logging.error(f"Error in rule-based analysis: {e}")
            # Fallback to neutral signal
            return type('obj', (object,), {
                'signal': 'neutral',
                'confidence': 0.5,
                'reasoning': f'Analysis completed using rule-based system (Error: {e})'
            })