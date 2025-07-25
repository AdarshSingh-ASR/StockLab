import React, { useState, useEffect } from 'react'
import { TrendingUp, TrendingDown, Activity, Globe, BarChart3, DollarSign, AlertTriangle, RefreshCw } from 'lucide-react'
import { Card, CardContent, CardHeader, CardTitle } from '../components/ui/Card'
import { Button } from '../components/ui/Button'
import { apiService } from '../services/api'
import { formatCurrency, formatPercentage, formatNumber } from '../lib/utils'

interface MarketSummary {
  sp500_change: number
  nasdaq_change: number
  vix: number
  market_regime: string
}

interface SectorPerformance {
  sector: string
  change: number
  volume: number
  momentum: number
}

interface MarketIndicator {
  name: string
  value: number
  change: number
  status: 'bullish' | 'bearish' | 'neutral'
}

interface EconomicData {
  indicator: string
  value: number
  previous: number
  change: number
  date: string
}

export const MarketData: React.FC = () => {
  const [marketSummary, setMarketSummary] = useState<MarketSummary | null>(null)
  const [sectorPerformance, setSectorPerformance] = useState<SectorPerformance[]>([])
  const [marketIndicators, setMarketIndicators] = useState<MarketIndicator[]>([])
  const [economicData, setEconomicData] = useState<EconomicData[]>([])
  const [isLoading, setIsLoading] = useState(true)
  const [lastUpdated, setLastUpdated] = useState<Date>(new Date())
  const [dataSource, setDataSource] = useState<'real' | 'fallback'>('real')

  const fetchMarketData = async () => {
    try {
      setIsLoading(true)
      const [summary, sectors, indicators, economic] = await Promise.all([
        apiService.getMarketSummary(),
        apiService.getSectorPerformance(),
        apiService.getMarketIndicators(),
        apiService.getEconomicData()
      ])
      
      setMarketSummary(summary)
      setSectorPerformance(sectors)
      setMarketIndicators(indicators)
      setEconomicData(economic)
      setLastUpdated(new Date())
      
      // Determine if we're using real data based on whether values are realistic
      // This is a simple heuristic - in a real app you'd want the API to return this info
      const isRealData = summary && 
        Math.abs(summary.sp500_change) < 50 && 
        Math.abs(summary.nasdaq_change) < 50 && 
        summary.vix > 0 && summary.vix < 100
      setDataSource(isRealData ? 'real' : 'fallback')
    } catch (error) {
      console.error('Error fetching market data:', error)
      setDataSource('fallback')
    } finally {
      setIsLoading(false)
    }
  }

  useEffect(() => {
    fetchMarketData()
    // Refresh every 5 minutes
    const interval = setInterval(fetchMarketData, 5 * 60 * 1000)
    return () => clearInterval(interval)
  }, [])

  const getMarketRegimeColor = (regime: string) => {
    switch (regime.toLowerCase()) {
      case 'bull market':
        return 'text-green-600 bg-green-100'
      case 'bear market':
        return 'text-red-600 bg-red-100'
      default:
        return 'text-yellow-600 bg-yellow-100'
    }
  }

  const getIndicatorStatus = (indicator: MarketIndicator) => {
    switch (indicator.status) {
      case 'bullish':
        return 'text-green-600'
      case 'bearish':
        return 'text-red-600'
      default:
        return 'text-gray-600'
    }
  }

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="flex items-center space-x-2">
          <RefreshCw className="w-6 h-6 animate-spin" />
          <span>Loading market data...</span>
        </div>
      </div>
    )
  }

  return (
    <div className="space-y-8 p-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold mb-2">Market Data</h1>
          <p className="text-muted-foreground mb-3">Real-time market information and economic indicators</p>
          <div className="flex items-center space-x-2">
            <div className={`inline-flex items-center px-2 py-1 rounded-full text-xs font-medium ${
              dataSource === 'real' 
                ? 'text-green-600 bg-green-100' 
                : 'text-orange-600 bg-orange-100'
            }`}>
              {dataSource === 'real' ? '🟢 Live Data' : '🟡 Fallback Data'}
            </div>
            {dataSource === 'fallback' && (
              <span className="text-xs text-muted-foreground">
                Using fallback data - check API connection
              </span>
            )}
          </div>
        </div>
        <div className="flex items-center space-x-4">
          <div className="text-sm text-muted-foreground">
            Last updated: {lastUpdated.toLocaleTimeString()}
          </div>
          <Button onClick={fetchMarketData} disabled={isLoading} variant="outline" size="sm">
            <RefreshCw className={`w-4 h-4 mr-2 ${isLoading ? 'animate-spin' : ''}`} />
            Refresh
          </Button>
        </div>
      </div>

      {/* Market Summary */}
      {marketSummary && (
        <div className="grid gap-6 md:grid-cols-2 lg:grid-cols-4">
          <Card className="p-6">
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-4">
              <CardTitle className="text-sm font-medium">S&P 500</CardTitle>
              <BarChart3 className="h-4 w-4 text-muted-foreground" />
            </CardHeader>
            <CardContent>
              <div className={`text-2xl font-bold ${marketSummary.sp500_change >= 0 ? 'text-green-600' : 'text-red-600'}`}>
                {formatPercentage(marketSummary.sp500_change)}
              </div>
              <p className="text-xs text-muted-foreground mt-1">
                {marketSummary.sp500_change >= 0 ? 'Gaining' : 'Declining'} today
              </p>
            </CardContent>
          </Card>

          <Card className="p-6">
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-4">
              <CardTitle className="text-sm font-medium">NASDAQ</CardTitle>
              <TrendingUp className="h-4 w-4 text-muted-foreground" />
            </CardHeader>
            <CardContent>
              <div className={`text-2xl font-bold ${marketSummary.nasdaq_change >= 0 ? 'text-green-600' : 'text-red-600'}`}>
                {formatPercentage(marketSummary.nasdaq_change)}
              </div>
              <p className="text-xs text-muted-foreground mt-1">
                {marketSummary.nasdaq_change >= 0 ? 'Gaining' : 'Declining'} today
              </p>
            </CardContent>
          </Card>

          <Card className="p-6">
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-4">
              <CardTitle className="text-sm font-medium">VIX</CardTitle>
              <Activity className="h-4 w-4 text-muted-foreground" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">{marketSummary.vix.toFixed(1)}</div>
              <p className="text-xs text-muted-foreground mt-1">
                {marketSummary.vix > 20 ? 'High' : marketSummary.vix > 15 ? 'Moderate' : 'Low'} volatility
              </p>
            </CardContent>
          </Card>

          <Card className="p-6">
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-4">
              <CardTitle className="text-sm font-medium">Market Regime</CardTitle>
              <Globe className="h-4 w-4 text-muted-foreground" />
            </CardHeader>
            <CardContent>
              <div className={`inline-flex items-center px-2 py-1 rounded-full text-sm font-medium ${getMarketRegimeColor(marketSummary.market_regime)}`}>
                {marketSummary.market_regime}
              </div>
            </CardContent>
          </Card>
        </div>
      )}

      {/* Sector Performance */}
      <Card className="p-6">
        <CardHeader className="pb-6">
          <CardTitle className="flex items-center space-x-2 text-xl">
            <BarChart3 className="h-5 w-5" />
            <span>Sector Performance</span>
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
            {sectorPerformance.map((sector) => (
              <div key={sector.sector} className="flex items-center justify-between p-4 border rounded-lg hover:bg-gray-50 transition-colors">
                <div>
                  <p className="font-medium text-sm">{sector.sector}</p>
                  <p className="text-xs text-muted-foreground mt-1">
                    Volume: {formatNumber(sector.volume)}
                  </p>
                </div>
                <div className="text-right">
                  <div className={`text-lg font-bold ${sector.change >= 0 ? 'text-green-600' : 'text-red-600'}`}>
                    {formatPercentage(sector.change)}
                  </div>
                  <div className={`text-sm ${sector.momentum >= 0 ? 'text-green-600' : 'text-red-600'}`}>
                    {sector.momentum >= 0 ? '↗' : '↘'} {Math.abs(sector.momentum).toFixed(1)}
                  </div>
                </div>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>

      {/* Market Indicators and Economic Data */}
      <div className="grid gap-8 md:grid-cols-2">
        <Card className="p-6">
          <CardHeader className="pb-6">
            <CardTitle className="flex items-center space-x-2 text-xl">
              <Activity className="h-5 w-5" />
              <span>Market Indicators</span>
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-6">
              {marketIndicators.map((indicator) => (
                <div key={indicator.name} className="flex items-center justify-between p-3 border rounded-lg">
                  <div>
                    <p className="font-medium text-sm">{indicator.name}</p>
                    <p className="text-xs text-muted-foreground mt-1">
                      {indicator.value.toFixed(2)}
                    </p>
                  </div>
                  <div className="text-right">
                    <div className={`font-bold text-sm ${getIndicatorStatus(indicator)}`}>
                      {formatPercentage(indicator.change)}
                    </div>
                    <div className={`text-xs mt-1 ${indicator.status === 'bullish' ? 'text-green-600' : indicator.status === 'bearish' ? 'text-red-600' : 'text-gray-600'}`}>
                      {indicator.status.charAt(0).toUpperCase() + indicator.status.slice(1)}
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>

        {/* Economic Data */}
        <Card className="p-6">
          <CardHeader className="pb-6">
            <CardTitle className="flex items-center space-x-2 text-xl">
              <DollarSign className="h-5 w-5" />
              <span>Economic Indicators</span>
            </CardTitle>
            <p className="text-xs text-muted-foreground mt-2">
              Note: Economic data uses realistic values based on current conditions. Real-time economic data requires specialized APIs.
            </p>
          </CardHeader>
          <CardContent>
            <div className="space-y-6">
              {economicData.map((data) => (
                <div key={data.indicator} className="flex items-center justify-between p-3 border rounded-lg">
                  <div>
                    <p className="font-medium text-sm">{data.indicator}</p>
                    <p className="text-xs text-muted-foreground mt-1">
                      {data.date}
                    </p>
                  </div>
                  <div className="text-right">
                    <div className="font-bold text-sm">
                      {data.value.toFixed(2)}
                    </div>
                    <div className={`text-xs mt-1 ${data.change >= 0 ? 'text-green-600' : 'text-red-600'}`}>
                      {data.change >= 0 ? '+' : ''}{data.change.toFixed(2)} vs prev
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Market Alerts */}
      <Card className="p-6">
        <CardHeader className="pb-6">
          <CardTitle className="flex items-center space-x-2 text-xl">
            <AlertTriangle className="h-5 w-5" />
            <span>Market Alerts</span>
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            {marketSummary && marketSummary.vix > 25 && (
              <div className="flex items-center space-x-3 p-4 bg-red-50 border border-red-200 rounded-lg">
                <AlertTriangle className="h-4 w-4 text-red-600" />
                <span className="text-red-800 text-sm">High volatility detected (VIX: {marketSummary.vix.toFixed(1)})</span>
              </div>
            )}
            {marketSummary && marketSummary.sp500_change < -2 && (
              <div className="flex items-center space-x-3 p-4 bg-yellow-50 border border-yellow-200 rounded-lg">
                <AlertTriangle className="h-4 w-4 text-yellow-600" />
                <span className="text-yellow-800 text-sm">Significant market decline detected</span>
              </div>
            )}
            {marketSummary && marketSummary.sp500_change > 2 && (
              <div className="flex items-center space-x-3 p-4 bg-green-50 border border-green-200 rounded-lg">
                <TrendingUp className="h-4 w-4 text-green-600" />
                <span className="text-green-800 text-sm">Strong market rally in progress</span>
              </div>
            )}
            {(!marketSummary || (marketSummary.vix <= 25 && Math.abs(marketSummary.sp500_change) <= 2)) && (
              <div className="text-center text-muted-foreground py-8">
                <div className="text-sm">No active market alerts</div>
                <div className="text-xs mt-1">Market conditions are stable</div>
              </div>
            )}
          </div>
        </CardContent>
      </Card>
    </div>
  )
} 