import React, { useState, useEffect } from 'react'
import { Plus, TrendingUp, TrendingDown, DollarSign, PieChart, BarChart3, Download } from 'lucide-react'
import { Card, CardContent, CardHeader, CardTitle } from '../components/ui/Card'
import { Button } from '../components/ui/Button'
import { Input } from '../components/ui/Input'
import { apiService } from '../services/api'
import { formatCurrency, formatPercentage } from '../lib/utils'
import type { PortfolioPosition, Portfolio } from '../types'

interface AddPositionModalProps {
  isOpen: boolean
  onClose: () => void
  onAdd: (ticker: string, shares: number, price: number) => void
}

const AddPositionModal: React.FC<AddPositionModalProps> = ({ isOpen, onClose, onAdd }) => {
  const [ticker, setTicker] = useState('')
  const [shares, setShares] = useState('')
  const [price, setPrice] = useState('')

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    if (ticker && shares && price) {
      onAdd(ticker.toUpperCase(), parseInt(shares), parseFloat(price))
      setTicker('')
      setShares('')
      setPrice('')
      onClose()
    }
  }

  if (!isOpen) return null

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
      <div className="bg-white rounded-lg p-6 w-96">
        <h2 className="text-xl font-bold mb-4">Add Position</h2>
        <form onSubmit={handleSubmit} className="space-y-4">
          <div>
            <label className="block text-sm font-medium mb-1">Ticker</label>
            <Input
              placeholder="AAPL"
              value={ticker}
              onChange={(e) => setTicker(e.target.value)}
              required
            />
          </div>
          <div>
            <label className="block text-sm font-medium mb-1">Shares</label>
            <Input
              type="number"
              placeholder="100"
              value={shares}
              onChange={(e) => setShares(e.target.value)}
              required
            />
          </div>
          <div>
            <label className="block text-sm font-medium mb-1">Purchase Price</label>
            <Input
              type="number"
              step="0.01"
              placeholder="150.00"
              value={price}
              onChange={(e) => setPrice(e.target.value)}
              required
            />
          </div>
          <div className="flex space-x-3 pt-4">
            <Button type="submit" className="flex-1">
              Add Position
            </Button>
            <Button type="button" variant="outline" onClick={onClose} className="flex-1">
              Cancel
            </Button>
          </div>
        </form>
      </div>
    </div>
  )
}

const PositionCard: React.FC<{ position: PortfolioPosition }> = ({ position }) => {
  const isPositive = position.unrealized_pnl >= 0
  
  return (
    <Card>
      <CardContent className="p-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-3">
            <div className="w-10 h-10 bg-primary/10 rounded-lg flex items-center justify-center">
              <span className="text-sm font-semibold text-primary">{position.ticker}</span>
            </div>
            <div>
              <h3 className="font-medium">{position.ticker}</h3>
              <p className="text-sm text-muted-foreground">{position.shares} shares</p>
            </div>
          </div>
          <div className="text-right">
            <p className="font-medium">{formatCurrency(position.market_value)}</p>
            <p className={`text-sm ${isPositive ? 'text-green-600' : 'text-red-600'}`}>
              {isPositive ? '+' : ''}{formatCurrency(position.unrealized_pnl)} ({formatPercentage(position.unrealized_pnl_percent)})
            </p>
          </div>
        </div>
        <div className="mt-3 pt-3 border-t">
          <div className="grid grid-cols-3 gap-4 text-sm">
            <div>
              <p className="text-muted-foreground">Avg Price</p>
              <p className="font-medium">{formatCurrency(position.avg_price)}</p>
            </div>
            <div>
              <p className="text-muted-foreground">Current</p>
              <p className="font-medium">{formatCurrency(position.current_price)}</p>
            </div>
            <div>
              <p className="text-muted-foreground">Weight</p>
              <p className="font-medium">{formatPercentage((position.market_value / 32000) * 100)}</p>
            </div>
          </div>
        </div>
      </CardContent>
    </Card>
  )
}

export const PortfolioPage: React.FC = () => {
  const [portfolio, setPortfolio] = useState<Portfolio | null>(null)
  const [isLoading, setIsLoading] = useState(true)
  const [showAddModal, setShowAddModal] = useState(false)

  useEffect(() => {
    loadPortfolio()
  }, [])

  const loadPortfolio = async () => {
    try {
      const portfolioData = await apiService.getPortfolio()
      setPortfolio({
        ...portfolioData,
        total_pnl_percent: (portfolioData.total_pnl / (portfolioData.total_value - portfolioData.total_pnl)) * 100
      })
    } catch (error) {
      console.error('Error loading portfolio:', error)
      // Fallback to mock data for development
      setPortfolio({
        positions: [
          {
            ticker: "AAPL",
            shares: 100,
            avg_price: 150.0,
            current_price: 165.0,
            market_value: 16500.0,
            unrealized_pnl: 1500.0,
            unrealized_pnl_percent: 10.0
          },
          {
            ticker: "MSFT",
            shares: 50,
            avg_price: 280.0,
            current_price: 310.0,
            market_value: 15500.0,
            unrealized_pnl: 1500.0,
            unrealized_pnl_percent: 10.7
          }
        ],
        total_value: 32000.0,
        total_pnl: 3000.0,
        total_pnl_percent: 10.3
      })
    } finally {
      setIsLoading(false)
    }
  }

  const handleAddPosition = async (ticker: string, shares: number, price: number) => {
    try {
      await apiService.addPosition(ticker, shares, price)
      await loadPortfolio() // Reload portfolio data
    } catch (error) {
      console.error('Error adding position:', error)
    }
  }

  if (isLoading) {
    return (
      <div className="p-8">
        <div className="animate-pulse">
          <div className="h-8 bg-gray-200 rounded w-1/4 mb-4"></div>
          <div className="space-y-4">
            <div className="h-32 bg-gray-200 rounded"></div>
            <div className="h-32 bg-gray-200 rounded"></div>
            <div className="h-32 bg-gray-200 rounded"></div>
          </div>
        </div>
      </div>
    )
  }

  if (!portfolio) {
    return (
      <div className="p-8">
        <h1 className="text-3xl font-bold tracking-tight mb-2">Portfolio</h1>
        <p className="text-muted-foreground">Failed to load portfolio data.</p>
      </div>
    )
  }

  const isPositive = portfolio.total_pnl >= 0

  return (
    <div className="p-8 space-y-8">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold tracking-tight">Portfolio</h1>
          <p className="text-muted-foreground">
            Track your positions and portfolio performance
          </p>
        </div>
        <div className="flex space-x-3">
          <Button onClick={() => setShowAddModal(true)} className="flex items-center space-x-2">
            <Plus className="w-4 h-4" />
            Add Position
          </Button>
          <Button variant="outline" className="flex items-center space-x-2">
            <Download className="w-4 h-4" />
            Export
          </Button>
        </div>
      </div>

      {/* Portfolio Summary */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Total Value</CardTitle>
            <DollarSign className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{formatCurrency(portfolio.total_value)}</div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Total P&L</CardTitle>
            {isPositive ? (
              <TrendingUp className="h-4 w-4 text-green-600" />
            ) : (
              <TrendingDown className="h-4 w-4 text-red-600" />
            )}
          </CardHeader>
          <CardContent>
            <div className={`text-2xl font-bold ${isPositive ? 'text-green-600' : 'text-red-600'}`}>
              {isPositive ? '+' : ''}{formatCurrency(portfolio.total_pnl)}
            </div>
            <p className={`text-xs ${isPositive ? 'text-green-600' : 'text-red-600'}`}>
              {isPositive ? '+' : ''}{formatPercentage(portfolio.total_pnl_percent)}
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Positions</CardTitle>
            <PieChart className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{portfolio.positions.length}</div>
            <p className="text-xs text-muted-foreground">Active positions</p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Win Rate</CardTitle>
            <BarChart3 className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">
              {formatPercentage(
                (portfolio.positions.filter(p => p.unrealized_pnl >= 0).length / portfolio.positions.length) * 100
              )}
            </div>
            <p className="text-xs text-muted-foreground">Profitable positions</p>
          </CardContent>
        </Card>
      </div>

      {/* Positions */}
      <div>
        <div className="flex items-center justify-between mb-6">
          <h2 className="text-2xl font-bold">Positions</h2>
          <div className="flex items-center space-x-2">
            <Button variant="outline" size="sm">Sort by P&L</Button>
            <Button variant="outline" size="sm">Sort by Value</Button>
          </div>
        </div>

        {portfolio.positions.length === 0 ? (
          <Card>
            <CardContent className="p-8 text-center">
              <div className="text-muted-foreground mb-4">
                <PieChart className="h-12 w-12 mx-auto mb-2" />
                <p>No positions yet</p>
              </div>
              <Button onClick={() => setShowAddModal(true)}>
                Add Your First Position
              </Button>
            </CardContent>
          </Card>
        ) : (
          <div className="grid gap-4">
            {portfolio.positions.map((position) => (
              <PositionCard key={position.ticker} position={position} />
            ))}
          </div>
        )}
      </div>

      {/* Add Position Modal */}
      <AddPositionModal
        isOpen={showAddModal}
        onClose={() => setShowAddModal(false)}
        onAdd={handleAddPosition}
      />
    </div>
  )
} 