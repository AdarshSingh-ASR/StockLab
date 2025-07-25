"""
Modular ML Model Benchmark Runner with W&B Integration
=====================================================

A rapid, one-shot ML development script for benchmarking multiple models
with Weights & Biases integration and LLM-generated reports.

Example
-------
$ python benchmark_models.py --experiment-run "model_comparison_v1" --models Tempus_v2 Tempus_v3 --years 1
"""
import argparse
import logging
import inspect
import os
import sys
import time
import json
import importlib.util
from datetime import date, datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

import quantstats_lumi as qs
from tqdm import tqdm
from Components.RuleBasedAnalysis import StatisticalReportGenerator
import quantstats_lumi as qs

# Optional dependencies
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    wandb = None

from Components.TickerData import TickerData
from Components.BackTesting import CustomBacktestingEngine
from Components.alpha_pipeline import AlphaVectorPipeline
from Components.WandbReportGenerator import WandbReportGenerator

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)s | %(message)s")


class ModelBenchmarkRunner:
    """Modular benchmark runner for ML models with W&B integration"""

    def __init__(self, experiment_run: str, models: List[str], days: int = 252,
                 out_dir: str = "benchmark_results", sample_size: int = 1200,
                 prediction_window: int = 3, run_name: str = None,
                 use_wandb: bool = False, use_statistical_reports: bool = False):
        self.experiment_run = experiment_run
        self.models = models
        self.days = days
        self.sample_size = sample_size #sample_size
        self.prediction_window = prediction_window
        self.run_name = run_name or f"benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(exist_ok=True)
        self.cache_dir = Path(f"{self.out_dir}/data_cache")
        self.cache_dir.mkdir(exist_ok=True)
        self.riskFreeRate = 0.04348 # 10yr U.S. Treasury Yield
        self.initial_capital = 10000.0
        self.risk_aversion = 0.8
        self.max_position_pct = 0.25
        self.transaction_cost_bps = 5.0

        # Initialize W&B
        self.use_wandb = use_wandb
        if self.use_wandb:
            self.wandb_run = None
            self.initialize_wandb()

        # Initialize statistical report generator
        self.use_statistical_reports = use_statistical_reports
        if self.use_statistical_reports:
            self.report_generator = StatisticalReportGenerator()
        else:
            self.report_generator = None

        # Model results storage
        self.model_results = {}

        # Data caching
        self.stock_data = None
        self.benchmark_returns = None
        self.stock_data_cache_path = self.cache_dir / "stock_data_cache.parquet"
        self.benchmark_data_cache_path = self.cache_dir / "benchmark_data_cache.parquet"
        self.alpha_data_cache_path = self.cache_dir / "alpha_data_cache.parquet"

    def initialize_wandb(self):
        """Initialize Weights & Biases - find existing run by name instead of creating new one"""
        if not WANDB_AVAILABLE:
            logging.warning("⚠️  W&B not available (install with: pip install wandb)")
            return

        try:
            # Search for existing run by name
            api = wandb.Api()
            runs = api.runs("taltmann0818-wake-forest-university/tft-us-equities")

            existing_run = None
            for run in runs:
                if self.run_name in run.name or run.name == self.run_name:
                    existing_run = run
                    break

            if existing_run:
                # Use existing run - create a new wandb session that logs to the existing run
                self.wandb_run = wandb.init(
                    project="tft-us-equities",
                    id=existing_run.id,
                    resume="allow",
                    config={
                        "backtest_days": self.days,
                        "backtest_sample_size": self.sample_size,
                        "backtest_initial_capital": self.initial_capital,
                        "backtest_risk_aversion": self.risk_aversion,
                        "backtest_max_position_pct": self.max_position_pct,
                        "backtest_transaction_cost_bps": self.transaction_cost_bps
                    }
                )
                logging.info("✅ Found and resumed existing W&B run: %s (ID: %s)", self.run_name, existing_run.id)
            else:
                # Create new run if not found
                self.wandb_run = wandb.init(
                    project="tft-us-equities",
                    name=self.run_name,
                    config={
                        "backtest_days": self.days,
                        "backtest_sample_size": self.sample_size,
                        "backtest_initial_capital": self.initial_capital,
                        "backtest_risk_aversion": self.risk_aversion,
                        "backtest_max_position_pct": self.max_position_pct,
                        "backtest_transaction_cost_bps": self.transaction_cost_bps
                    }
                )
                logging.info("✅ Created new W&B run: %s", self.run_name)

        except Exception as e:
            logging.warning("⚠️  Could not initialize W&B: %s", e)

    def initialize_statistical_reports(self):
        """Initialize statistical report generator"""
        try:
            self.report_generator = StatisticalReportGenerator()
            logging.info("✅ Initialized statistical report generator")
        except Exception as e:
            logging.warning("⚠️  Could not initialize report generator: %s", e)

    def load_model_inference(self, model_name: str):
        """Dynamically load model inference class"""
        model_dir = Path("Models") / model_name
        inference_path = model_dir / "inference.py"

        if not inference_path.exists():
            raise FileNotFoundError(f"Inference script not found: {inference_path}")

        # Load the inference module
        spec = importlib.util.spec_from_file_location(f"{model_name}_inference", inference_path)
        inference_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(inference_module)
        # Get the inference class (assumes naming convention)
        class_name = f"{model_name.replace('_', '').replace('.', '')}Inference"
        if hasattr(inference_module, class_name):
            return getattr(inference_module, class_name)()
        else:
            # Try common naming patterns
            for attr_name in dir(inference_module):
                attr = getattr(inference_module, attr_name)
                if (hasattr(attr, '__class__') and
                    hasattr(attr, 'predict') and
                    'Inference' in attr.__class__.__name__):
                    return attr

        raise AttributeError(f"Could not find inference class in {inference_path}")

    def load_model_datamodule(self, model_name: str, config: dict):
        model_dir = Path("Models") / model_name
        datamodule_path = model_dir / "datamodule.py"

        if not datamodule_path.exists():
            logging.warning("⚠️  No datamodule found for %s", model_name)
            return None

        spec = importlib.util.spec_from_file_location("datamodule", datamodule_path)
        datamodule_module = importlib.util.module_from_spec(spec)

        try:
            spec.loader.exec_module(datamodule_module)
        except Exception as e:
            logging.error("❌  Import error in %s: %s", datamodule_path, e)
            return None          # bail early if the file itself is broken

        # ── search for a suitable class ────────────────────────────────────────────
        for obj in datamodule_module.__dict__.values():
            if (inspect.isclass(obj)
                    and 'DataModule' in obj.__name__
                    and hasattr(obj, 'prepare_data')):
                try:
                    return obj(                      # instantiate it
                        config=config,
                        days=self.days,
                        use_cache=True,
                        sample_size=self.sample_size,
                        cache_dir=self.cache_dir
                    )
                except Exception as e:
                    logging.error("❌  Failed to instantiate %s: %s", obj.__name__, e)
                    return None

        # fell through the loop → nothing matched
        logging.warning("⚠️  No DataModule subclass found in %s", datamodule_path)
        return None

    def _load_stock_data_from_cache(self) -> Optional[pd.DataFrame]:
        """Load stock data from cache if available and valid"""
        if not self.stock_data_cache_path.exists():
            return None

        try:
            cached_data = pd.read_parquet(self.stock_data_cache_path)
            logging.info("✅ Loaded stock data from cache: %s", self.stock_data_cache_path)
            return cached_data
        except Exception as e:
            logging.warning("⚠️  Failed to load stock data cache: %s", e)
            return None

    def _save_stock_data_to_cache(self, data: pd.DataFrame):
        """Save stock data to cache"""
        try:
            data.to_parquet(self.stock_data_cache_path)
            logging.info("💾 Saved stock data to cache: %s", self.stock_data_cache_path)
        except Exception as e:
            logging.warning("⚠️  Failed to save stock data cache: %s", e)

    def _load_benchmark_data_from_cache(self) -> Optional[pd.DataFrame]:
        """Load benchmark data from cache if available and valid"""
        if not self.benchmark_data_cache_path.exists():
            return None

        try:
            cached_data = pd.read_parquet(self.benchmark_data_cache_path)
            logging.info("✅ Loaded benchmark data from cache: %s", self.benchmark_data_cache_path)
            return cached_data
        except Exception as e:
            logging.warning("⚠️  Failed to load benchmark data cache: %s", e)
            return None

    def _save_benchmark_data_to_cache(self, data: pd.DataFrame):
        """Save benchmark data to cache"""
        try:
            data.to_parquet(self.benchmark_data_cache_path)
            logging.info("💾 Saved benchmark data to cache: %s", self.benchmark_data_cache_path)
        except Exception as e:
            logging.warning("⚠️  Failed to save benchmark data cache: %s", e)

    def _save_alpha_data_to_cache(self, data: pd.DataFrame):
        """Save benchmark data to cache"""
        try:
            data.to_parquet(self.alpha_data_cache_path)
            logging.info("💾 Saved Alpha data to cache: %s", self.alpha_data_cache_path)
        except Exception as e:
            logging.warning("⚠️  Failed to save Alpha data cache: %s", e)

    def prepare_benchmark_data(self):
        """Prepare benchmark data (NDX index) with caching"""
        # Try to load from cache first
        self.benchmark_returns = self._load_benchmark_data_from_cache()
        if self.benchmark_returns is not None:
            return

        try:
            # Use TickerData to get benchmark data
            data_retriever = TickerData(
                indicator_list=None,
                days=self.days,
                prediction_mode=True
            )

            benchmark_prices = data_retriever.get_ohlc_for_ticker('I:NDX').reset_index()
            if benchmark_prices is not None and 'Close' in benchmark_prices.columns:
                index_returns = data_retriever.get_ohlc_for_ticker('I:NDX')
                index_returns = index_returns['Close'].pct_change().dropna()
                index_returns.index = index_returns.index.tz_localize(None)
                index_returns.name = "daily_return"
                self.benchmark_returns = index_returns
                logging.info("✅ Loaded benchmark data (NDX)")

                # Save to cache
                self._save_benchmark_data_to_cache(pd.DataFrame(self.benchmark_returns))
            else:
                logging.warning("⚠️  Could not load benchmark data")
                self.benchmark_data = None
        except Exception as e:
            logging.error("❌ Error loading benchmark data: %s", e)
            self.benchmark_data = None

    def run_model_backtest(self, model_name: str, inference_class, data: pd.DataFrame) -> Dict[str, Any]:
        """Run backtest for a single model using custom backtesting engine"""
        logging.info("🔮 Running backtest for %s", model_name)

        # Run inference to get predictions
        predictions = inference_class.predict(data)

        # Process predictions to generate alpha signals
        alpha_dict, alpha_df = AlphaVectorPipeline(polygon_api_key='XizU4KyrwjCA6bxHrR5_eQnUxwFFUnI2').run(predictions)
        logging.info(f"Generated alpha signals for {len(alpha_dict)} dates")

        # Save to cache
        self._save_alpha_data_to_cache(alpha_df)

        if not alpha_dict:
            logging.warning("⚠️  No alpha signals generated for %s", model_name)
            return None

        # Initialize custom backtesting engine
        backtesting_engine = CustomBacktestingEngine(
            initial_capital=self.initial_capital,
            risk_aversion=self.risk_aversion,
            max_position_pct=self.max_position_pct,
            transaction_cost_bps=self.transaction_cost_bps
        )

        # Run backtest using alpha signals
        try:
            returns_df = backtesting_engine.run_backtest(
                alpha_dict=alpha_dict,
                stock_data=self.stock_data,
            )

            if returns_df.empty:
                logging.warning("⚠️  No returns generated for %s", model_name)
                return None

            # Calculate performance metrics using quantstats_lumi
            strategy_returns = returns_df.set_index('date')['daily_return']

            # Calculate comprehensive metrics using quantstats_lumi
            metrics = self._calculate_quantstats_metrics(strategy_returns, self.benchmark_returns, model_name)

            return {
                'results_df': pd.DataFrame([metrics]),
                'returns_df': returns_df,
                'strategy_returns': returns_df,
                'trade_summary': backtesting_engine.trade_log(),
                'alpha_signals': alpha_dict
            }

        except Exception as e:
            logging.error(f"Error running backtest for {model_name}: {e}")
            return None

    def _calculate_quantstats_metrics(self, strat_returns: pd.Series, index_returns: pd.Series, model_name: str) -> Dict[str, float]:
        """
        Calculate comprehensive performance metrics using quantstats_lumi.
        """

        import quantstats_lumi as qs

        # Calculate standalone metrics using FULL strategy returns
        full_total_return = qs.stats.comp(strat_returns)
        full_cagr = qs.stats.cagr(strat_returns)
        full_sharpe = qs.stats.sharpe(strat_returns)
        full_sortino = qs.stats.sortino(strat_returns)
        full_max_dd = qs.stats.max_drawdown(strat_returns)
        value_at_risk = qs.stats.value_at_risk(strat_returns)

        # For benchmark-relative metrics, use aligned data
        common_start = max(strat_returns.index.min(), index_returns.index.min())
        common_end = min(strat_returns.index.max(), index_returns.index.max())

        strategy_aligned = strat_returns[
            (strat_returns.index >= common_start) &
            (strat_returns.index <= common_end)
            ]

        benchmark_aligned = index_returns[
            (index_returns.index >= common_start) &
            (index_returns.index <= common_end)
            ]

        # Calculate benchmark-relative metrics
        metrics = np.array(qs.reports.metrics(strategy_aligned, benchmark_aligned, mode='full', rf=0.042, display=False))

        # Calculate basic metrics
        metrics = {
            'model': model_name,
            'backtesting_date': date.today(),
            'total_return': float(full_total_return)*100,
            'cagr': full_cagr,
            'sharpe_ratio': full_sharpe,
            'sortino_ratio': full_sortino,
            'max_drawdown': full_max_dd,
            'dVaR': value_at_risk,
            'Alpha': float(metrics[58][1]),
            'Beta': float(metrics[57][1])
        }

        return metrics

    def create_comparison_plots(self) -> Dict[str, Any]:
        """Create comparison plots for all models using Plotly"""
        plots = {}

        series_dict: Dict[str, pd.Series] = {}
        # --- benchmark ----------------------------------------------------
        bench = self.benchmark_returns
        if isinstance(bench, pd.DataFrame):
            if {"date", "daily_return"}.issubset(bench.columns):
                bench = bench.set_index("date")["daily_return"]
            else:
                raise ValueError(
                    "benchmark_returns DataFrame must have ['date', 'daily_return'] columns"
                )
        elif isinstance(bench, (list, tuple)):
            bench = pd.Series(bench)

        if not isinstance(bench, pd.Series):
            raise ValueError(
                "`self.benchmark_returns` must be a Series, list/tuple, or a "
                "DataFrame with ['date','daily_return'].  Got type "
                f"{type(self.benchmark_returns)}"
            )

        series_dict["NDX Benchmark"] = (1 + bench).cumprod() - 1

        # --- models -------------------------------------------------------
        for model_name, res in self.model_results.items():
            if (
                res
                and "strategy_returns" in res
                and not res["strategy_returns"].empty
                and {"date", "daily_return"}.issubset(res["strategy_returns"].columns)
            ):
                s = res["strategy_returns"].set_index("date")["daily_return"]
                series_dict[model_name] = (1 + s).cumprod() - 1

        if len(series_dict) <= 1:
            raise ValueError("No valid model/benchmark return series to plot.")

        # ------------------------------------------------------------------
        # 2. Align on the same date index & drop NaNs
        # ------------------------------------------------------------------
        cum_df = (
            pd.concat(series_dict, axis=1)
            .sort_index()        # chronological
            .dropna(how="all")   # drop rows where *every* col is NaN
            .ffill()             # forward-fill gaps (line charts hate NaNs)
        )

        # Create Plotly performance comparison chart
        fig_performance = go.Figure()

        # Plot each series
        for col in cum_df.columns:
            fig_performance.add_trace(go.Scatter(
                x=cum_df.index,
                y=cum_df[col].values,
                mode='lines',
                name=col,
                line=dict(width=2),
                hovertemplate='<b>%{fullData.name}</b><br>' +
                             'Date: %{x}<br>' +
                             'Return: %{y:.2%}<br>' +
                             '<extra></extra>'
            ))

        fig_performance.update_layout(
            title="Model vs Benchmark Cumulative Return",
            xaxis_title="Date",
            yaxis_title="Cumulative Return",
            hovermode='x unified',
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01
            )
        )

        # Export performance chart to HTML
        perf_html_path = os.path.join(self.out_dir, "performance_comparison.html")
        fig_performance.write_html(perf_html_path)

        # Log to wandb directly with the Plotly figure
        if self.wandb_run:
            self.wandb_run.log({"performance_comparison": wandb.Plotly(fig_performance)})

        plots['performance_comparison'] = fig_performance

        # Metrics comparison using Plotly
        metrics_data = []
        for model_name, results in self.model_results.items():
            if results and 'results_df' in results and not results['results_df'].empty:
                model_metrics = results['results_df'].copy()
                model_metrics['model'] = model_name
                metrics_data.append(model_metrics)

        if metrics_data:
            all_metrics = pd.concat(metrics_data, ignore_index=True)

            # Handle potential missing columns gracefully
            sharpe_col = 'sharpe_ratio' if 'sharpe_ratio' in all_metrics.columns else 'Sharpe'
            return_col = 'total_return' if 'total_return' in all_metrics.columns else 'Total Return'
            drawdown_col = 'max_drawdown' if 'max_drawdown' in all_metrics.columns else 'Max Drawdown'
            alpha_col = 'Alpha' if 'Alpha' in all_metrics.columns else 'alpha'

            # Create Plotly subplots for metrics comparison
            fig_metrics = make_subplots(
                rows=2, cols=2,
                subplot_titles=['Sharpe Ratio', 'Total Return', 'Max Drawdown', 'Alpha']
            )

            if sharpe_col in all_metrics.columns:
                fig_metrics.add_trace(
                    go.Bar(x=all_metrics['model'], y=all_metrics[sharpe_col], 
                          name='Sharpe', marker_color='navy', opacity=0.7),
                    row=1, col=1
                )

            if return_col in all_metrics.columns:
                fig_metrics.add_trace(
                    go.Bar(x=all_metrics['model'], y=all_metrics[return_col], 
                          name='Return', marker_color='green', opacity=0.7),
                    row=1, col=2
                )

            if drawdown_col in all_metrics.columns:
                fig_metrics.add_trace(
                    go.Bar(x=all_metrics['model'], y=all_metrics[drawdown_col], 
                          name='Drawdown', marker_color='red', opacity=0.7),
                    row=2, col=1
                )

            if alpha_col in all_metrics.columns:
                alpha_values = all_metrics[alpha_col].fillna(0)
                fig_metrics.add_trace(
                    go.Bar(x=all_metrics['model'], y=alpha_values, 
                          name='Alpha', marker_color='orange', opacity=0.7),
                    row=2, col=2
                )

            fig_metrics.update_layout(
                title='Model Metrics Comparison',
                height=600,
                showlegend=False
            )

            # Export metrics chart to HTML
            metrics_html_path = os.path.join(self.out_dir, "metrics_comparison.html")
            fig_metrics.write_html(metrics_html_path)

            # Log to wandb directly with the Plotly figure
            if self.wandb_run:
                self.wandb_run.log({"metrics_comparison": wandb.Plotly(fig_metrics)})

            plots['metrics_comparison'] = fig_metrics

        return plots

    def generate_llm_report(self, summary_data: Dict[str, Any]) -> str:
        """Generate statistical analysis report using rule-based system"""
        if not self.use_statistical_reports:
            return "Report generation not available (Statistical analysis not configured)"

        try:
            # Use statistical report generator instead of LLM
            performance_report = StatisticalReportGenerator.generate_performance_report(summary_data)
            recommendations = StatisticalReportGenerator.generate_recommendations(summary_data)
            
            # Combine reports
            full_report = {
                "overall_recommendation": "ADOPT" if any(
                    results.get('sharpe_ratio', 0) > 0.5 
                    for results in summary_data.get('model_results', {}).values()
                ) else "DISCARD",
                "exec_summary": f"Analysis of {len(summary_data.get('model_results', {}))} models over {self.days} days",
                "performance_analysis": performance_report,
                "risk_assessment": "Risk assessment based on Sharpe ratios and drawdowns",
                "recommendations": [
                    "Monitor market regime changes",
                    "Consider transaction costs in live trading", 
                    "Diversify across multiple strategies",
                    "Regular rebalancing recommended"
                ]
            }
            
            return full_report

        except Exception as e:
            logging.error("❌ Error generating statistical report: %s", e)
            return f"Error generating statistical report: {e}"

    def publish_wandb_report(self, plots: Dict[str, Any], llm_report: str):
        """Publish comprehensive enterprise-grade report to W&B using WandbReportGenerator"""
        if not WANDB_AVAILABLE or not self.wandb_run:
            logging.info("⚠️  W&B not available, skipping report publishing")
            return

        try:
            # Initialize the WandbReportGenerator
            report_generator = WandbReportGenerator(
                project_name="tft-us-equities",
                run_name=self.run_name
            )
            # Create the Weights & Biases report
            report = report_generator.create_wandb_report(
                llm_summary=llm_report,
                benchmark_plots=plots,
                wandb_run=self.wandb_run
            )

            if report:
                logging.info(f"✅ W&B report created and saved to run {self.run_name}")

                # Log summary metrics to W&B run
                summary_metrics = {}
                for model_name, results in self.model_results.items():
                    if results and 'results_df' in results and not results['results_df'].empty:
                        model_summary = results['results_df'].groupby('model').agg({
                            'sharpe_ratio': 'mean',
                            'total_return': 'mean',
                            'max_drawdown': 'mean'
                        }).to_dict('records')[0]

                        for metric, value in model_summary.items():
                            summary_metrics[f"{model_name}_{metric}"] = value

                #self.wandb_run.log(summary_metrics)

            else:
                logging.error("❌ Failed to create W&B report")

        except Exception as e:
            logging.error("❌ Error publishing W&B report: %s", e)

    def run_benchmark(self):
        """Run the complete benchmark process"""
        logging.info("🚀 Starting benchmark run: %s", self.experiment_run)

        # Prepare benchmark data
        self.prepare_benchmark_data()

        # Prepare the initial sample set with caching
        self.stock_data = self._load_stock_data_from_cache()
        if self.stock_data is None:
            data_retriever = TickerData(
                indicator_list=None,
                days=self.days,
                prediction_window=self.prediction_window,
                prediction_mode=True,
                sample_size=self.sample_size
            )
            self.stock_data = data_retriever.preprocess_data().reset_index()
            logging.info("✅ Finished pulling initial OHLCV data shared among models")

            # Save to cache
            self._save_stock_data_to_cache(self.stock_data)
        else:
            logging.info("✅ Using cached OHLCV data shared among models")

        # Run each model
        trade_histories = []
        for model_name in self.models:
            try:
                logging.info("📊 Processing model: %s", model_name)

                # Load model components
                inference_class = self.load_model_inference(model_name)
                datamodule = self.load_model_datamodule(model_name, inference_class.constants)

                # Prepare data
                if datamodule:
                    data = datamodule.prepare_data(self.stock_data)

                    if data.empty:
                        logging.warning("⚠️  No data available for %s", model_name)
                        continue
                else:
                    logging.warning("⚠️  No datamodule available for %s", model_name)
                    continue

                # Run backtest
                results = self.run_model_backtest(model_name, inference_class, data)
                self.model_results[model_name] = results
                trade_histories.append(results['trade_summary'])

                logging.info("✅ Completed %s", model_name)

            except Exception as e:
                logging.error("❌ Error processing %s: %s", model_name, e)
                self.model_results[model_name] = None

        # Ensure output directory exists
        os.makedirs(self.out_dir, exist_ok=True)

        # Generate comparison plots (Plotly plots are already saved as HTML in the method)
        plots = self.create_comparison_plots()
        logging.info("💾 Plotly plots saved to %s directory", self.out_dir)

        # Generate LLM report
        summary_data = {
            model_name: {
                'sharpe_ratio': results['results_df']['sharpe_ratio'] if results and 'results_df' in results and not results['results_df'].empty else 0,
                'sortino_ratio': results['results_df']['sortino_ratio'] if results and 'results_df' in results and not
results['results_df'].empty else 0,
                'total_return': results['results_df']['total_return'] if results and 'results_df' in results and not results['results_df'].empty else 0,
                'cagr': results['results_df']['cagr'] if results and 'results_df' in results and not
results['results_df'].empty else 0,
                'max_drawdown': results['results_df']['max_drawdown'] if results and 'results_df' in results and not results['results_df'].empty else 0,
                'alpha': results['results_df']['Alpha'] if results and 'results_df' in results and not
                results['results_df'].empty else 0,
                'beta': results['results_df']['Beta'] if results and 'results_df' in results and not
results['results_df'].empty else 0,
                'dvar': results['results_df']['dVaR'] if results and 'results_df' in results and not
results['results_df'].empty else 0
            }
            for model_name, results in self.model_results.items()
        }

        llm_report = self.generate_llm_report(summary_data)

        # Publish to W&B
        if self.use_wandb:
            self.publish_wandb_report(plots, llm_report)

        # Save summary
        summary_path = self.out_dir / "benchmark_summary.json"
        with open(summary_path, 'w') as f:
            json.dump({
                'experiment_run': self.experiment_run,
                'models': self.models,
                'summary_data': summary_data,
                'timestamp': datetime.now().isoformat()
            }, f, indent=2, default=str)

        # Save trade history
        pd.concat(trade_histories).to_parquet(self.out_dir / "trade_history.parquet", index=False)

        logging.info("✅ Benchmark complete! Results saved to %s", self.out_dir)

        if self.wandb_run:
            self.wandb_run.finish()

def main():
    parser = argparse.ArgumentParser(description="Modular ML Model Benchmark Runner")
    parser.add_argument("--run-name", required=True,
                       help="W&B run name for the benchmark experiment")
    parser.add_argument("--models", nargs="+", required=True,
                       help="Model names to benchmark (e.g., Tempus_v2 Tempus_v3)")
    parser.add_argument("--days", type=int, default=252,
                       help="Days of data to use for backtesting")
    parser.add_argument("--horizon", default=3,
                   help="Forecast horizon for models")
    parser.add_argument("--sample-size", type=int, default=1000)
    parser.add_argument("--out-dir", default="benchmark_results",
                       help="Output directory for results")
    parser.add_argument("--use-llm", default=True, type=bool,
                       help="Use statistical analysis for generating reports")
    parser.add_argument("--use-reporting", default=True, type=bool,
                       help="Use W&B reporting for generating reports")

    args = parser.parse_args()

    # Run benchmark
    runner = ModelBenchmarkRunner(
        experiment_run=args.run_name,
        models=args.models,
        days=args.days,
        out_dir=args.out_dir,
        sample_size=args.sample_size,
        prediction_window=args.horizon,
        run_name=args.run_name,
        use_wandb = args.use_reporting,
        use_statistical_reports = args.use_llm
    )

    runner.run_benchmark()


if __name__ == "__main__":
    t0 = time.time()
    main()
    logging.info("Total runtime %.1f s", time.time() - t0)
