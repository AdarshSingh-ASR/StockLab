from Agents.aswath_damodaran import AswathDamodaranAgent
from Agents.ben_graham import BenGrahamAgent
from Agents.bill_ackman import BillAckmanAgent
from Agents.cathie_wood import CathieWoodAgent
from Agents.charlie_munger import CharlieMungerAgent
from Agents.peter_lynch import PeterLynchAgent
from Agents.phil_fisher import PhilFisherAgent
from Agents.stanley_druckenmiller import StanleyDruckenmillerAgent
from Agents.warren_buffett import WarrenBuffettAgent
from Agents.valuation import ValuationAgent
from Agents.fundamentals import FundamentalsAgent

from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
import streamlit as st

from typing import List, Optional, Type, Union, Dict, Any

AgentType = Type  # or more strictly: Type[object]

class AgentManager:
    ALL_AGENTS: List[AgentType] = [
        AswathDamodaranAgent,
        BenGrahamAgent,
        BillAckmanAgent,
        CathieWoodAgent,
        CharlieMungerAgent,
        PeterLynchAgent,
        PhilFisherAgent,
        StanleyDruckenmillerAgent,
        WarrenBuffettAgent,
        ValuationAgent,
        FundamentalsAgent,
    ]
    AGENT_MAP: Dict[str, AgentType] = {cls.__name__: cls for cls in ALL_AGENTS}

    def __init__(
        self,
        metrics: pd.DataFrame,
        agents: Optional[List[Union[str, AgentType]]] = None,
        period: Optional[str] = 'Annual',
        streamlit_progress: Optional[bool] = False
    ):
        self.tickers: List[str] = list(metrics.ticker.values)
        self.metrics: pd.DataFrame = metrics

        if agents is None:
            self.agent_classes = self.ALL_AGENTS.copy()
        else:
            resolved: List[AgentType] = []
            for item in agents:
                if isinstance(item, str):
                    if item in self.AGENT_MAP:
                        resolved.append(self.AGENT_MAP[item])
                    else:
                        raise ValueError(
                            f"Unknown agent name '{item}'. Valid names: {list(self.AGENT_MAP.keys())}"
                        )
                elif isinstance(item, type):
                    resolved.append(item)
                else:
                    raise TypeError(
                        f"Agent entries must be class or string, got {type(item)}"
                    )
            self.agent_classes = resolved

        self.period = 'Q' if period == 'Quarterly' else 'FY'
        self.limit = 4 if period == 'Quarterly' else 10
        self.threshold_matrix_path = {'two_digit_sic':                ('../../Agents/Matrices/Fundamentals Matrix - 2digit SIC.csv'),
                                      'business_services_sic':        ('../../Agents/Matrices/Fundamentals Matrix - 4digit SIC 73 - Business Services.csv')
                                     }
        self.streamlit_progress = streamlit_progress

    def _analyze_one_ticker(self, ticker: str) -> Dict[str, Any]:
        """
        Run every agent for this one ticker,
        return a map agent_name → result (or exception).
        """
        results: Dict[str, Any] = {}
        print(f"Analyzing ticker: {ticker}")
        for AgentCls in self.agent_classes:
            name = AgentCls.__name__
            print(f"Running agent: {name}")
            try:
                agent = AgentCls(ticker, self.metrics, analysis_period=self.period, analysis_limit=self.limit, threshold_matrix_path=self.threshold_matrix_path)
                print(f"Agent {name} created successfully")
                result = agent.analyze()
                print(f"Agent {name} result: {result}")
                results[name] = result
            except Exception as e:
                print(f"Agent {name} failed with error: {e}")
                results[name] = e
        print(f"All results for {ticker}: {results}")
        return results

    def _summarize(self, raw: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
        """
        Given the nested dict { ticker: {agent_name: result, …}, … },
        build a DataFrame with counts of each signal and total score.
        """
        rows = []
        for ticker, agents_out in raw.items():
            bullish = bearish = neutral = 0
            total_score = 0.0
            for res in agents_out.values():
                # skip agents that raised exceptions
                if isinstance(res, Exception):
                    continue
                sig = res.get('signal', None)
                if sig == 'bullish':
                    bullish += 1
                elif sig == 'bearish':
                    bearish += 1
                else:
                    neutral += 1
                # assume score is numeric, defaulting to 0
                total_score += float(res.get('score', 0))

                #pe_ratio = res.get('pe_ratio', None) if res.get('name') == 'Fundamentals' else None
                #pb_ratio = res.get('pb_ratio', None) if res.get('name') == 'Fundamentals' else None
                #ps_ratio = res.get('ps_ratio', None) if res.get('name') == 'Fundamentals' else None

            filtered = self.metrics[self.metrics['ticker'] == ticker]
            #evEBITDA = filtered['enterprise_value'].iloc[0] / filtered['ebitda'].iloc[0]

            rows.append({
                'Ticker': ticker,
                'Company Name': filtered['company_name'].iloc[0],
                'Bullish': bullish,
                'Bearish': bearish,
                'Neutral': neutral,
                'Score': total_score,
                #'Price/Earnings': pe_ratio,
                #'Price/Book': pb_ratio,
                #'Price/Sales': ps_ratio,
                #'Enterprise Value/EBITDA': evEBITDA,
            })

        df = pd.DataFrame(rows)
        df['Signal'] = df[['Bullish', 'Bearish', 'Neutral']].idxmax(axis=1)
        df = df[['Ticker', 'Company Name', 'Signal', 'Score', 'Bullish', 'Bearish', 'Neutral']]#, 'Price/Earnings', 'Price/Book', 'Price/Sales', 'Enterprise Value/EBITDA']]
        return df.set_index('Ticker')

    def agent_analysis(self) -> Dict[str, Dict[str, Any]]:
        """
        Parallelize over tickers: each thread runs _analyze_one_ticker.
        Returns:
          { ticker1: {AgentA: res, AgentB: res, …}, ticker2: {…}, … }
        """
        final_results: Dict[str, Dict[str, Any]] = {}
        # Reduce max workers for faster processing with fewer tickers
        max_workers = min(len(self.tickers), 4)  # Reduced from 50 to 4
        total = len(self.tickers)
        processed = 0
        # set up Streamlit progress bar if requested
        if self.streamlit_progress:
            progress_bar = st.progress(0)
            status_text = st.empty()

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_ticker = {
                executor.submit(self._analyze_one_ticker, tk): tk
                for tk in self.tickers
            }

            for future in as_completed(future_to_ticker):
                tk = future_to_ticker[future]
                try:
                    final_results[tk] = future.result()
                except Exception as e:
                    print(f"Error analyzing {tk}: {e}")
                    # Provide fallback result instead of error
                    final_results[tk] = {}
                    for AgentCls in self.agent_classes:
                        name = AgentCls.__name__
                        final_results[tk][name] = {
                            "name": name.replace("Agent", ""),
                            "signal": "neutral",
                            "score": 5.0,
                            "confidence": 0.5,
                            "reasoning": f"Analysis failed: {str(e)}"
                        }

                if self.streamlit_progress:
                    processed += 1
                    # update progress bar as a percentage
                    progress_bar.progress(int(processed / total * 100))
                    status_text.text(f"{round(processed / total * 100,2)}% completed")
                    
        if self.streamlit_progress:
            progress_bar.empty()
            status_text.empty()
        summary_df = self._summarize(final_results)
        return final_results, summary_df