"""
Fundamentals Module
==================
Provides functions for searching and analyzing fundamental financial data.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

def search_line_items(
    ticker: str,
    line_items: List[str],
    period: str = 'Annual',
    limit: int = 5,
    df: Optional[pd.DataFrame] = None
) -> Tuple[pd.DataFrame, str]:
    """
    Search for financial line items in the metrics dataframe.
    
    Args:
        ticker: Stock ticker symbol
        line_items: List of financial metrics to search for
        period: Analysis period ('Annual' or 'Quarterly')
        limit: Number of periods to retrieve
        df: DataFrame containing financial metrics
        
    Returns:
        Tuple of (financial_line_items_df, sic_code)
    """
    try:
        if df is None or df.empty:
            # Return empty DataFrame with requested columns if no data
            empty_data = {item: [0.0] * limit for item in line_items}
            empty_df = pd.DataFrame(empty_data)
            return empty_df, "0000"  # Default SIC code
        
        # Filter data for the specific ticker (handle both 'Ticker' and 'ticker' column names)
        ticker_column = 'Ticker' if 'Ticker' in df.columns else 'ticker'
        ticker_data = df[df[ticker_column] == ticker].copy()
        
        if ticker_data.empty:
            # Return empty DataFrame if ticker not found
            empty_data = {item: [0.0] * limit for item in line_items}
            empty_df = pd.DataFrame(empty_data)
            return empty_df, "0000"
        
        # Extract SIC code (assuming it's in the data)
        sic_code = str(ticker_data.get('SIC_Code', '0000').iloc[0]) if 'SIC_Code' in ticker_data.columns else "0000"
        
        # Create result DataFrame with requested line items
        result_data = {}
        for item in line_items:
            if item in ticker_data.columns:
                # Get the value, handle different data types
                value = ticker_data[item].iloc[0]
                if isinstance(value, (list, np.ndarray)):
                    # If it's already a list/array, use it
                    result_data[item] = value[:limit]
                else:
                    # If it's a single value, create a list
                    result_data[item] = [value] * limit
            else:
                # If item not found, use zeros
                result_data[item] = [0.0] * limit
        
        result_df = pd.DataFrame(result_data)
        return result_df, sic_code
        
    except Exception as e:
        logger.error(f"Error in search_line_items for {ticker}: {e}")
        # Return empty DataFrame on error
        empty_data = {item: [0.0] * limit for item in line_items}
        empty_df = pd.DataFrame(empty_data)
        return empty_df, "0000"

def get_metric_value(
    threshold_matrix: pd.DataFrame,
    sic_code: str,
    metric_name: str
) -> float:
    """
    Get the threshold value for a specific metric and SIC code.
    
    Args:
        threshold_matrix: DataFrame containing threshold values
        sic_code: SIC code for the industry
        metric_name: Name of the metric to look up
        
    Returns:
        Threshold value for the metric
    """
    try:
        if threshold_matrix is None or threshold_matrix.empty:
            return 0.0
        
        # Try to find the metric in the threshold matrix
        if metric_name in threshold_matrix.columns:
            # If SIC code exists in the data, use it; otherwise use first row
            if 'SIC_Code' in threshold_matrix.columns:
                sic_data = threshold_matrix[threshold_matrix['SIC_Code'] == sic_code]
                if not sic_data.empty:
                    return float(sic_data[metric_name].iloc[0])
            
            # Fallback to first row
            return float(threshold_matrix[metric_name].iloc[0])
        
        # Default thresholds for common metrics
        default_thresholds = {
            'return_on_equity': 0.15,      # 15% ROE
            'net_margin': 0.10,            # 10% net margin
            'operating_margin': 0.15,      # 15% operating margin
            'revenue_growth_qoq': 0.05,    # 5% quarterly revenue growth
            'eps_growth_qoq': 0.05,        # 5% quarterly EPS growth
            'bookValue_growth_qoq': 0.05,  # 5% quarterly book value growth
            'current_ratio': 1.5,          # 1.5 current ratio
            'debt_to_equity': 0.5,         # 0.5 debt-to-equity
            'free_cash_flow_per_share': 0.0,  # Positive FCF
            'price_to_book': 1.5,          # 1.5 P/B ratio
            'price_to_earnings': 15.0,     # 15 P/E ratio
            'ev_to_ebitda': 10.0,          # 10 EV/EBITDA
            'ev_to_ebit': 12.0,            # 12 EV/EBIT
        }
        
        return default_thresholds.get(metric_name, 0.0)
        
    except Exception as e:
        logger.error(f"Error in get_metric_value for {metric_name}: {e}")
        return 0.0

def calculate_financial_ratios(
    financial_data: pd.DataFrame
) -> Dict[str, float]:
    """
    Calculate common financial ratios from financial data.
    
    Args:
        financial_data: DataFrame containing financial metrics
        
    Returns:
        Dictionary of calculated ratios
    """
    ratios = {}
    
    try:
        if financial_data.empty:
            return ratios
        
        # Extract values (assuming first row contains current data)
        data = financial_data.iloc[0] if len(financial_data) > 0 else financial_data
        
        # Calculate ratios if data is available
        if 'net_income' in data and 'outstanding_shares' in data:
            if data['outstanding_shares'] > 0:
                ratios['eps'] = data['net_income'] / data['outstanding_shares']
        
        if 'book_value' in data and 'outstanding_shares' in data:
            if data['outstanding_shares'] > 0:
                ratios['book_value_per_share'] = data['book_value'] / data['outstanding_shares']
        
        if 'market_cap' in data and 'revenue' in data:
            if data['revenue'] > 0:
                ratios['price_to_sales'] = data['market_cap'] / data['revenue']
        
        if 'market_cap' in data and 'net_income' in data:
            if data['net_income'] > 0:
                ratios['price_to_earnings'] = data['market_cap'] / data['net_income']
        
        if 'market_cap' in data and 'book_value' in data:
            if data['book_value'] > 0:
                ratios['price_to_book'] = data['market_cap'] / data['book_value']
        
        if 'debt_to_equity' in data:
            ratios['debt_to_equity'] = data['debt_to_equity']
        
        if 'current_ratio' in data:
            ratios['current_ratio'] = data['current_ratio']
        
        if 'return_on_equity' in data:
            ratios['return_on_equity'] = data['return_on_equity']
        
        if 'operating_margin' in data:
            ratios['operating_margin'] = data['operating_margin']
        
    except Exception as e:
        logger.error(f"Error calculating financial ratios: {e}")
    
    return ratios

def validate_financial_data(
    financial_data: pd.DataFrame,
    required_columns: List[str]
) -> bool:
    """
    Validate that financial data contains required columns and valid values.
    
    Args:
        financial_data: DataFrame to validate
        required_columns: List of required column names
        
    Returns:
        True if data is valid, False otherwise
    """
    try:
        if financial_data is None or financial_data.empty:
            return False
        
        # Check if required columns exist
        missing_columns = [col for col in required_columns if col not in financial_data.columns]
        if missing_columns:
            logger.warning(f"Missing required columns: {missing_columns}")
            return False
        
        # Check for non-null values in required columns
        for col in required_columns:
            if financial_data[col].isnull().all():
                logger.warning(f"Column {col} contains only null values")
                return False
        
        return True
        
    except Exception as e:
        logger.error(f"Error validating financial data: {e}")
        return False 