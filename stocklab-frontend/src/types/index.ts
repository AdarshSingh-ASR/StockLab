export interface StockData {
  ticker: string
  company_name: string
  open: number
  high: number
  low: number
  close: number
  volume: number
  date: string
}

export interface AgentAnalysis {
  name: string
  signal: 'bullish' | 'bearish' | 'neutral'
  score: number
  confidence: number
  reasoning: string
  fundamental_analysis?: {
    score: number
    details: string
  }
  consistency_analysis?: {
    score: number
    details: string
  }
  moat_analysis?: {
    score: number
    details: string
  }
  management_analysis?: {
    score: number
    details: string
  }
  intrinsic_value_analysis?: {
    intrinsic_value: number
    margin_of_safety: number
  }
}

export interface StockAnalysis {
  ticker: string
  company_name: string
  signal: 'bullish' | 'bearish' | 'neutral'
  score: number
  bullish: number
  bearish: number
  neutral: number
  agent_analyses: Record<string, AgentAnalysis>
}

export interface ModelPrediction {
  ticker: string
  predicted_price: number
  confidence: number
  date: string
}

export interface TechnicalIndicators {
  ema_20: number
  ema_50: number
  ema_100: number
  stoch_rsi14: number
  macd: number
  hmm_state: number
  [key: string]: number
}

export interface FundamentalMetrics {
  pe_ratio: number
  pb_ratio: number
  ps_ratio: number
  ev_ebitda: number
  return_on_equity: number
  debt_to_equity: number
  operating_margin: number
  current_ratio: number
  [key: string]: number
}

export interface Agent {
  name: string
  description: string
  icon: string
  isActive: boolean
}

export interface AnalysisRequest {
  tickers: string[]
  agents: string[]
  period: 'Annual' | 'Quarterly'
  include_technical: boolean
  include_fundamentals: boolean
  include_predictions: boolean
}

export interface AnalysisResponse {
  summary: StockAnalysis[]
  detailed_analyses: Record<string, AgentAnalysis[]>
  predictions?: ModelPrediction[]
  technical_data?: Record<string, TechnicalIndicators>
  fundamental_data?: Record<string, FundamentalMetrics>
}

export interface PortfolioPosition {
  ticker: string
  shares: number
  avg_price: number
  current_price: number
  market_value: number
  unrealized_pnl: number
  unrealized_pnl_percent: number
}

export interface Portfolio {
  positions: PortfolioPosition[]
  total_value: number
  total_pnl: number
  total_pnl_percent: number
} 