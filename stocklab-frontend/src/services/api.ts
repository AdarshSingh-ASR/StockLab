import type { 
  AnalysisRequest, 
  AnalysisResponse, 
  StockData, 
  ModelPrediction,
  TechnicalIndicators,
  FundamentalMetrics 
} from '../types'

const API_BASE_URL = 'http://localhost:8000' // Assuming FastAPI backend

class ApiService {
  private async request<T>(endpoint: string, options?: RequestInit): Promise<T> {
    const url = `${API_BASE_URL}${endpoint}`
    const response = await fetch(url, {
      headers: {
        'Content-Type': 'application/json',
        ...options?.headers,
      },
      ...options,
    })

    if (!response.ok) {
      throw new Error(`API request failed: ${response.statusText}`)
    }

    return response.json()
  }

  // Stock Analysis
  async analyzeStocks(request: AnalysisRequest): Promise<AnalysisResponse> {
    return this.request<AnalysisResponse>('/api/analyze', {
      method: 'POST',
      body: JSON.stringify(request),
    })
  }

  // Get stock data
  async getStockData(ticker: string, days: number = 365): Promise<StockData[]> {
    return this.request<StockData[]>(`/api/stock-data/${ticker}?days=${days}`)
  }

  // Get model predictions
  async getPredictions(tickers: string[]): Promise<ModelPrediction[]> {
    return this.request<ModelPrediction[]>('/api/predictions', {
      method: 'POST',
      body: JSON.stringify({ tickers }),
    })
  }

  // Get technical indicators
  async getTechnicalIndicators(ticker: string): Promise<TechnicalIndicators> {
    return this.request<TechnicalIndicators>(`/api/technical/${ticker}`)
  }

  // Get fundamental metrics
  async getFundamentalMetrics(ticker: string): Promise<FundamentalMetrics> {
    return this.request<FundamentalMetrics>(`/api/fundamentals/${ticker}`)
  }

  // Get available agents
  async getAgents(): Promise<string[]> {
    return this.request<string[]>('/api/agents')
  }

  // Get agent performance data
  async getAgentPerformance(): Promise<Record<string, {
    win_rate: number
    avg_return: number
    sharpe_ratio: number
    max_drawdown: number
    total_trades: number
    confidence: number
    last_analysis: string
    is_active: boolean
  }>> {
    return this.request('/api/agents/performance')
  }

  // Get market data summary
  async getMarketSummary(): Promise<{
    sp500_change: number
    nasdaq_change: number
    vix: number
    market_regime: string
  }> {
    return this.request('/api/market-summary')
  }

  // Get sector performance data
  async getSectorPerformance(): Promise<Array<{
    sector: string
    change: number
    volume: number
    momentum: number
  }>> {
    return this.request('/api/sector-performance')
  }

  // Get market indicators
  async getMarketIndicators(): Promise<Array<{
    name: string
    value: number
    change: number
    status: 'bullish' | 'bearish' | 'neutral'
  }>> {
    return this.request('/api/market-indicators')
  }

  // Get economic data
  async getEconomicData(): Promise<Array<{
    indicator: string
    value: number
    previous: number
    change: number
    date: string
  }>> {
    return this.request('/api/economic-data')
  }

  // Portfolio management
  async getPortfolio(): Promise<{
    positions: any[]
    total_value: number
    total_pnl: number
  }> {
    return this.request('/api/portfolio')
  }

  async addPosition(ticker: string, shares: number, price: number): Promise<void> {
    return this.request('/api/portfolio/positions', {
      method: 'POST',
      body: JSON.stringify({ ticker, shares, price }),
    })
  }

  // Backtesting
  async runBacktest(strategy: string, tickers: string[], startDate: string, endDate: string): Promise<{
    returns: number[]
    sharpe_ratio: number
    max_drawdown: number
    win_rate: number
    num_trades: number
    avg_positions: number
    total_return: number
    volatility: number
  }> {
    return this.request('/api/backtest', {
      method: 'POST',
      body: JSON.stringify({ strategy, tickers, startDate, endDate }),
    })
  }
}

export const apiService = new ApiService() 