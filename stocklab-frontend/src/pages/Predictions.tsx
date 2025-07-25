import React, { useState, useEffect } from 'react'
import { Search, TrendingUp, TrendingDown, Target, BarChart3, Calendar, Download, RefreshCw } from 'lucide-react'
import { Card, CardContent, CardHeader, CardTitle } from '../components/ui/Card'
import { Button } from '../components/ui/Button'
import { Input } from '../components/ui/Input'
import { apiService } from '../services/api'
import { formatCurrency, formatPercentage } from '../lib/utils'
import type { ModelPrediction } from '../types'

interface PredictionCardProps {
  prediction: ModelPrediction
  currentPrice?: number
}

const PredictionCard: React.FC<PredictionCardProps> = ({ prediction, currentPrice }) => {
  const priceChange = currentPrice ? prediction.predicted_price - currentPrice : 0
  const priceChangePercent = currentPrice ? (priceChange / currentPrice) * 100 : 0
  const isPositive = priceChange >= 0

  const getConfidenceColor = (confidence: number) => {
    if (confidence >= 0.8) return 'text-green-600'
    if (confidence >= 0.6) return 'text-yellow-600'
    return 'text-red-600'
  }

  const getConfidenceLabel = (confidence: number) => {
    if (confidence >= 0.8) return 'High'
    if (confidence >= 0.6) return 'Medium'
    return 'Low'
  }

  return (
    <Card>
      <CardContent className="p-6">
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center space-x-3">
            <div className="w-12 h-12 bg-primary/10 rounded-lg flex items-center justify-center">
              <span className="text-lg font-semibold text-primary">{prediction.ticker}</span>
            </div>
            <div>
              <h3 className="text-lg font-semibold">{prediction.ticker}</h3>
              <p className="text-sm text-muted-foreground">Predicted for {prediction.date}</p>
            </div>
          </div>
          <div className="text-right">
            <div className={`inline-flex items-center px-3 py-1 rounded-full text-sm font-medium border ${
              isPositive ? 'border-green-200 bg-green-50 text-green-700' : 'border-red-200 bg-red-50 text-red-700'
            }`}>
              {isPositive ? <TrendingUp className="w-4 h-4 mr-1" /> : <TrendingDown className="w-4 h-4 mr-1" />}
              {isPositive ? 'BULLISH' : 'BEARISH'}
            </div>
          </div>
        </div>

        <div className="grid grid-cols-2 gap-4 mb-4">
          <div>
            <p className="text-sm text-muted-foreground">Predicted Price</p>
            <p className="text-2xl font-bold text-primary">{formatCurrency(prediction.predicted_price)}</p>
          </div>
          <div>
            <p className="text-sm text-muted-foreground">Confidence</p>
            <p className={`text-2xl font-bold ${getConfidenceColor(prediction.confidence)}`}>
              {formatPercentage(prediction.confidence * 100)}
            </p>
            <p className={`text-xs ${getConfidenceColor(prediction.confidence)}`}>
              {getConfidenceLabel(prediction.confidence)} Confidence
            </p>
          </div>
        </div>

        {currentPrice && (
          <div className="space-y-2">
            <div className="flex justify-between text-sm">
              <span className="text-muted-foreground">Current Price:</span>
              <span className="font-medium">{formatCurrency(currentPrice)}</span>
            </div>
            <div className="flex justify-between text-sm">
              <span className="text-muted-foreground">Expected Change:</span>
              <span className={`font-medium ${isPositive ? 'text-green-600' : 'text-red-600'}`}>
                {isPositive ? '+' : ''}{formatCurrency(priceChange)} ({isPositive ? '+' : ''}{formatPercentage(priceChangePercent)})
              </span>
            </div>
          </div>
        )}

        <div className="mt-4 pt-4 border-t">
          <div className="flex items-center justify-between text-sm">
            <span className="text-muted-foreground">Model Used:</span>
            <span className="font-medium">Tempus v3.0</span>
          </div>
        </div>
      </CardContent>
    </Card>
  )
}

interface PredictionChartProps {
  ticker: string
  predictions: ModelPrediction[]
}

const PredictionChart: React.FC<PredictionChartProps> = ({ ticker, predictions }) => {
  // This would integrate with a charting library like Recharts
  // For now, showing a simple representation
  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center space-x-2">
          <BarChart3 className="w-5 h-5" />
          <span>Price Forecast - {ticker}</span>
        </CardTitle>
      </CardHeader>
      <CardContent>
        <div className="h-64 bg-gray-50 rounded-lg flex items-center justify-center">
          <div className="text-center text-muted-foreground">
            <BarChart3 className="w-12 h-12 mx-auto mb-2" />
            <p>Price chart integration coming soon</p>
            <p className="text-sm">Will show historical prices and predictions</p>
          </div>
        </div>
      </CardContent>
    </Card>
  )
}

export const PredictionsPage: React.FC = () => {
  const [tickers, setTickers] = useState('')
  const [predictions, setPredictions] = useState<ModelPrediction[]>([])
  const [isLoading, setIsLoading] = useState(false)
  const [selectedTicker, setSelectedTicker] = useState<string>('')
  const [currentPrices, setCurrentPrices] = useState<Record<string, number>>({})

  const handleGetPredictions = async () => {
    if (!tickers.trim()) return

    setIsLoading(true)
    try {
      const tickerList = tickers.split(',').map(t => t.trim().toUpperCase()).filter(t => t)
      const predictionResults = await apiService.getPredictions(tickerList)
      setPredictions(predictionResults)
      
      // Get current prices for comparison (mock data for now)
      const mockCurrentPrices: Record<string, number> = {}
      tickerList.forEach(ticker => {
        mockCurrentPrices[ticker] = ticker === 'AAPL' ? 165.0 : 
                                   ticker === 'MSFT' ? 310.0 : 
                                   ticker === 'GOOGL' ? 140.0 : 
                                   ticker === 'AMZN' ? 180.0 : 100.0
      })
      setCurrentPrices(mockCurrentPrices)
      
    } catch (error) {
      console.error('Error getting predictions:', error)
    } finally {
      setIsLoading(false)
    }
  }

  const handleTickerClick = (ticker: string) => {
    setSelectedTicker(ticker)
  }

  const getPredictionForTicker = (ticker: string) => {
    return predictions.find(p => p.ticker === ticker)
  }

  return (
    <div className="p-8 space-y-8">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold tracking-tight">Predictions</h1>
          <p className="text-muted-foreground">
            ML-powered price predictions and market forecasting
          </p>
        </div>
        <div className="flex space-x-3">
          <Button 
            onClick={handleGetPredictions} 
            disabled={isLoading || !tickers.trim()}
            className="flex items-center space-x-2"
          >
            {isLoading ? (
              <RefreshCw className="w-4 h-4 animate-spin" />
            ) : (
              <Target className="w-4 h-4" />
            )}
            {isLoading ? 'Getting Predictions...' : 'Get Predictions'}
          </Button>
          <Button variant="outline" className="flex items-center space-x-2">
            <Download className="w-4 h-4" />
            Export
          </Button>
        </div>
      </div>

      {/* Input Section */}
      <Card>
        <CardHeader>
          <CardTitle>Prediction Configuration</CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <div>
            <Input
              label="Stock Tickers"
              placeholder="AAPL, MSFT, GOOGL (comma separated)"
              value={tickers}
              onChange={(e) => setTickers(e.target.value)}
            />
            <p className="text-xs text-muted-foreground mt-1">
              Enter stock tickers to get ML model predictions
            </p>
          </div>
          
          <div className="flex items-center space-x-4">
            <div className="flex items-center space-x-2">
              <Calendar className="w-4 h-4 text-muted-foreground" />
              <span className="text-sm text-muted-foreground">Prediction Horizon:</span>
              <select className="text-sm border rounded px-2 py-1">
                <option>1 Day</option>
                <option>1 Week</option>
                <option>1 Month</option>
                <option>3 Months</option>
              </select>
            </div>
            <div className="flex items-center space-x-2">
              <BarChart3 className="w-4 h-4 text-muted-foreground" />
              <span className="text-sm text-muted-foreground">Model:</span>
              <select className="text-sm border rounded px-2 py-1">
                <option>Tempus v3.0</option>
                <option>Tempus v2.0</option>
                <option>ResMLP</option>
              </select>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Predictions Grid */}
      {predictions.length > 0 && (
        <div className="space-y-6">
          <div className="flex items-center justify-between">
            <h2 className="text-2xl font-bold">Price Predictions</h2>
            <div className="flex items-center space-x-2">
              <Button variant="outline" size="sm">Sort by Confidence</Button>
              <Button variant="outline" size="sm">Sort by Change</Button>
            </div>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {predictions.map((prediction) => (
              <div 
                key={prediction.ticker} 
                onClick={() => handleTickerClick(prediction.ticker)}
                className="cursor-pointer hover:shadow-lg transition-shadow"
              >
                <PredictionCard 
                  prediction={prediction} 
                  currentPrice={currentPrices[prediction.ticker]}
                />
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Detailed Chart View */}
      {selectedTicker && getPredictionForTicker(selectedTicker) && (
        <div className="space-y-6">
          <div className="flex items-center justify-between">
            <h2 className="text-2xl font-bold">Detailed Analysis - {selectedTicker}</h2>
            <Button 
              variant="outline" 
              size="sm"
              onClick={() => setSelectedTicker('')}
            >
              Close
            </Button>
          </div>
          
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <PredictionChart 
              ticker={selectedTicker} 
              predictions={predictions.filter(p => p.ticker === selectedTicker)}
            />
            
            <Card>
              <CardHeader>
                <CardTitle>Model Insights</CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div>
                  <h4 className="font-medium mb-2">Technical Indicators</h4>
                  <div className="grid grid-cols-2 gap-2 text-sm">
                    <div className="flex justify-between">
                      <span className="text-muted-foreground">EMA 20:</span>
                      <span>${(currentPrices[selectedTicker] * 0.98).toFixed(2)}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-muted-foreground">EMA 50:</span>
                      <span>${(currentPrices[selectedTicker] * 0.95).toFixed(2)}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-muted-foreground">RSI:</span>
                      <span>65.2</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-muted-foreground">MACD:</span>
                      <span className="text-green-600">+0.85</span>
                    </div>
                  </div>
                </div>
                
                <div>
                  <h4 className="font-medium mb-2">Market Regime</h4>
                  <div className="inline-flex items-center px-3 py-1 rounded-full text-sm bg-green-100 text-green-700">
                    <div className="w-2 h-2 bg-green-500 rounded-full mr-2"></div>
                    Bull Market
                  </div>
                </div>
                
                <div>
                  <h4 className="font-medium mb-2">Risk Assessment</h4>
                  <div className="space-y-2">
                    <div className="flex justify-between text-sm">
                      <span className="text-muted-foreground">Volatility:</span>
                      <span className="text-yellow-600">Medium</span>
                    </div>
                    <div className="flex justify-between text-sm">
                      <span className="text-muted-foreground">Support Level:</span>
                      <span>${(currentPrices[selectedTicker] * 0.92).toFixed(2)}</span>
                    </div>
                    <div className="flex justify-between text-sm">
                      <span className="text-muted-foreground">Resistance Level:</span>
                      <span>${(currentPrices[selectedTicker] * 1.08).toFixed(2)}</span>
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>
        </div>
      )}

      {/* Empty State */}
      {predictions.length === 0 && !isLoading && (
        <Card>
          <CardContent className="p-12 text-center">
            <div className="text-muted-foreground mb-4">
              <Target className="h-16 w-16 mx-auto mb-4" />
              <h3 className="text-lg font-medium mb-2">No Predictions Yet</h3>
              <p>Enter stock tickers above to get ML-powered price predictions</p>
            </div>
            <div className="space-y-2 text-sm text-muted-foreground">
              <p>• Get predictions for multiple stocks at once</p>
              <p>• View confidence scores and price targets</p>
              <p>• Analyze technical indicators and market regimes</p>
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  )
} 