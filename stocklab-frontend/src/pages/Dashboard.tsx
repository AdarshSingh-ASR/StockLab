import React, { useState, useEffect } from 'react'
import { 
  TrendingUp, 
  TrendingDown, 
  DollarSign, 
  Activity,
  ArrowUpRight,
  ArrowDownRight
} from 'lucide-react'
import { Card, CardContent, CardHeader, CardTitle } from '../components/ui/Card'
import { formatCurrency, formatPercentage } from '../lib/utils'
import { apiService } from '../services/api'
import type { StockAnalysis } from '../types'

interface DashboardData {
  portfolio: {
    total_value: number
    total_pnl: number
  } | null
  marketSummary: {
    sp500_change: number
    nasdaq_change: number
    vix: number
    market_regime: string
  } | null
  topStocks: StockAnalysis[]
  isLoading: boolean
}

const StatCard: React.FC<{
  title: string
  value: string
  change: number
  icon: React.ReactNode
}> = ({ title, value, change, icon }) => (
  <Card>
    <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
      <CardTitle className="text-sm font-medium">{title}</CardTitle>
      {icon}
    </CardHeader>
    <CardContent>
      <div className="text-2xl font-bold">{value}</div>
      <div className="flex items-center text-xs text-muted-foreground">
        {change > 0 ? (
          <ArrowUpRight className="w-3 h-3 text-green-600 mr-1" />
        ) : (
          <ArrowDownRight className="w-3 h-3 text-red-600 mr-1" />
        )}
        {formatPercentage(Math.abs(change))} from last month
      </div>
    </CardContent>
  </Card>
)

export const Dashboard: React.FC = () => {
  const [data, setData] = useState<DashboardData>({
    portfolio: null,
    marketSummary: null,
    topStocks: [],
    isLoading: true
  })

  useEffect(() => {
    const loadDashboardData = async () => {
      try {
        // Load portfolio data
        const portfolioData = await apiService.getPortfolio()
        
        // Load market summary
        const marketData = await apiService.getMarketSummary()
        
        // Get top stock picks (using recent analysis results)
        // For now, we'll use a sample analysis of popular stocks
        const topStocksResponse = await apiService.analyzeStocks({
          tickers: ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA'],
          agents: ['WarrenBuffettAgent', 'PeterLynchAgent', 'CharlieMungerAgent'],
          period: 'Annual',
          include_technical: true,
          include_fundamentals: true,
          include_predictions: false
        })
        
        setData({
          portfolio: portfolioData,
          marketSummary: marketData,
          topStocks: topStocksResponse.summary.slice(0, 3), // Top 3 stocks
          isLoading: false
        })
      } catch (error) {
        console.error('Error loading dashboard data:', error)
        // Fallback to mock data if API fails
        setData({
          portfolio: { total_value: 125000, total_pnl: 8500 },
          marketSummary: {
            sp500_change: 0.85,
            nasdaq_change: 1.2,
            vix: 18.5,
            market_regime: 'Bull Market'
          },
          topStocks: [
            {
              ticker: 'AAPL',
              company_name: 'Apple Inc.',
              signal: 'bullish',
              score: 8.5,
              bullish: 7,
              bearish: 1,
              neutral: 2,
              agent_analyses: {}
            },
            {
              ticker: 'MSFT',
              company_name: 'Microsoft Corporation',
              signal: 'bullish',
              score: 8.2,
              bullish: 6,
              bearish: 2,
              neutral: 2,
              agent_analyses: {}
            },
            {
              ticker: 'GOOGL',
              company_name: 'Alphabet Inc.',
              signal: 'neutral',
              score: 6.8,
              bullish: 4,
              bearish: 3,
              neutral: 3,
              agent_analyses: {}
            }
          ],
          isLoading: false
        })
      }
    }

    loadDashboardData()
  }, [])

  if (data.isLoading) {
    return (
      <div className="p-8">
        <div className="animate-pulse">
          <div className="h-8 bg-gray-200 rounded w-1/4 mb-4"></div>
          <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4 mb-8">
            {[...Array(4)].map((_, i) => (
              <div key={i} className="h-32 bg-gray-200 rounded"></div>
            ))}
          </div>
          <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
            {[...Array(2)].map((_, i) => (
              <div key={i} className="h-64 bg-gray-200 rounded"></div>
            ))}
          </div>
        </div>
      </div>
    )
  }

  return (
    <div className="p-8 space-y-8">
      {/* Header */}
      <div>
        <h1 className="text-3xl font-bold tracking-tight">Dashboard</h1>
        <p className="text-muted-foreground">
          Welcome back! Here's what's happening with your portfolio and the market.
        </p>
      </div>

      {/* Stats Grid */}
      <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
        <StatCard
          title="Portfolio Value"
          value={formatCurrency(data.portfolio?.total_value || 0)}
          change={2.5}
          icon={<DollarSign className="h-4 w-4 text-muted-foreground" />}
        />
        <StatCard
          title="Total P&L"
          value={formatCurrency(data.portfolio?.total_pnl || 0)}
          change={6.8}
          icon={<TrendingUp className="h-4 w-4 text-muted-foreground" />}
        />
        <StatCard
          title="S&P 500"
          value={formatPercentage(data.marketSummary?.sp500_change || 0)}
          change={data.marketSummary?.sp500_change || 0}
          icon={<Activity className="h-4 w-4 text-muted-foreground" />}
        />
        <StatCard
          title="VIX"
          value={(data.marketSummary?.vix || 0).toString()}
          change={-5.2}
          icon={<TrendingDown className="h-4 w-4 text-muted-foreground" />}
        />
      </div>

      {/* Top Stocks */}
      <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
        <Card className="col-span-2">
          <CardHeader>
            <CardTitle>Top Stock Picks</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              {data.topStocks.map((stock) => (
                <div key={stock.ticker} className="flex items-center justify-between p-4 border rounded-lg">
                  <div className="flex items-center space-x-4">
                    <div className="w-10 h-10 bg-primary/10 rounded-lg flex items-center justify-center">
                      <span className="text-sm font-semibold text-primary">{stock.ticker}</span>
                    </div>
                    <div>
                      <p className="font-medium">{stock.company_name}</p>
                      <p className="text-sm text-muted-foreground">{stock.ticker}</p>
                    </div>
                  </div>
                  <div className="text-right">
                    <div className={`inline-flex items-center px-2 py-1 rounded-full text-xs font-medium ${
                      stock.signal === 'bullish' ? 'bg-green-100 text-green-800' :
                      stock.signal === 'bearish' ? 'bg-red-100 text-red-800' :
                      'bg-yellow-100 text-yellow-800'
                    }`}>
                      {stock.signal.toUpperCase()}
                    </div>
                    <p className="text-sm font-medium mt-1">Score: {stock.score}/10</p>
                  </div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Market Regime</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              <div className="flex items-center justify-between">
                <span className="text-sm font-medium">Current Regime</span>
                <span className="text-sm text-green-600 font-medium">{data.marketSummary?.market_regime || 'Unknown'}</span>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-sm font-medium">NASDAQ</span>
                <span className="text-sm text-green-600 font-medium">
                  +{formatPercentage(data.marketSummary?.nasdaq_change || 0)}
                </span>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-sm font-medium">Volatility</span>
                <span className="text-sm text-muted-foreground">{data.marketSummary?.vix || 0}</span>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Recent Activity */}
      <Card>
        <CardHeader>
          <CardTitle>Recent Activity</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            <div className="flex items-center space-x-4">
              <div className="w-2 h-2 bg-green-500 rounded-full"></div>
              <div className="flex-1">
                <p className="text-sm font-medium">AAPL analysis completed</p>
                <p className="text-xs text-muted-foreground">2 minutes ago</p>
              </div>
            </div>
            <div className="flex items-center space-x-4">
              <div className="w-2 h-2 bg-blue-500 rounded-full"></div>
              <div className="flex-1">
                <p className="text-sm font-medium">Portfolio rebalanced</p>
                <p className="text-xs text-muted-foreground">1 hour ago</p>
              </div>
            </div>
            <div className="flex items-center space-x-4">
              <div className="w-2 h-2 bg-yellow-500 rounded-full"></div>
              <div className="flex-1">
                <p className="text-sm font-medium">Market regime changed to {data.marketSummary?.market_regime || 'Bull Market'}</p>
                <p className="text-xs text-muted-foreground">3 hours ago</p>
              </div>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  )
} 