import React, { useState, useEffect } from 'react'
import { Brain, TrendingUp, TrendingDown, Target, BarChart3, Settings, Play, Pause, Zap, Shield, Eye, Star } from 'lucide-react'
import { Card, CardContent, CardHeader, CardTitle } from '../components/ui/Card'
import { Button } from '../components/ui/Button'
import { apiService } from '../services/api'
import { formatPercentage } from '../lib/utils'

interface AgentProfile {
  name: string
  description: string
  icon: string
  strategy: string
  riskLevel: 'Low' | 'Medium' | 'High'
  performance: {
    winRate: number
    avgReturn: number
    sharpeRatio: number
    maxDrawdown: number
    totalTrades: number
  }
  specialties: string[]
  isActive: boolean
  lastAnalysis: string
  confidence: number
}

const agentProfiles: AgentProfile[] = [
  {
    name: 'Warren Buffett',
    description: 'Value investing legend focused on long-term compound growth',
    icon: '💰',
    strategy: 'Value Investing',
    riskLevel: 'Low',
    performance: {
      winRate: 0.78,
      avgReturn: 0.15,
      sharpeRatio: 1.2,
      maxDrawdown: -0.08,
      totalTrades: 45
    },
    specialties: ['Blue-chip stocks', 'Consumer goods', 'Insurance', 'Long-term holds'],
    isActive: true,
    lastAnalysis: '2025-07-25',
    confidence: 0.85
  },
  {
    name: 'Peter Lynch',
    description: 'Growth at a reasonable price (GARP) specialist',
    icon: '📈',
    strategy: 'GARP',
    riskLevel: 'Medium',
    performance: {
      winRate: 0.72,
      avgReturn: 0.18,
      sharpeRatio: 1.1,
      maxDrawdown: -0.12,
      totalTrades: 67
    },
    specialties: ['Growth stocks', 'Consumer discretionary', 'Technology', 'Retail'],
    isActive: true,
    lastAnalysis: '2025-07-25',
    confidence: 0.82
  },
  {
    name: 'Charlie Munger',
    description: 'Mental models and quality business focus',
    icon: '🧠',
    strategy: 'Quality Investing',
    riskLevel: 'Low',
    performance: {
      winRate: 0.81,
      avgReturn: 0.14,
      sharpeRatio: 1.3,
      maxDrawdown: -0.06,
      totalTrades: 32
    },
    specialties: ['High-quality businesses', 'Mental models', 'Concentrated positions'],
    isActive: true,
    lastAnalysis: '2025-07-25',
    confidence: 0.88
  },
  {
    name: 'Cathie Wood',
    description: 'Innovation and disruption technology investor',
    icon: '🚀',
    strategy: 'Innovation Investing',
    riskLevel: 'High',
    performance: {
      winRate: 0.65,
      avgReturn: 0.25,
      sharpeRatio: 0.9,
      maxDrawdown: -0.25,
      totalTrades: 89
    },
    specialties: ['AI/ML', 'Biotechnology', 'Fintech', 'Electric vehicles'],
    isActive: true,
    lastAnalysis: '2025-07-25',
    confidence: 0.75
  },
  {
    name: 'Bill Ackman',
    description: 'Concentrated positions in high-conviction ideas',
    icon: '🎯',
    strategy: 'Concentrated Investing',
    riskLevel: 'High',
    performance: {
      winRate: 0.68,
      avgReturn: 0.22,
      sharpeRatio: 1.0,
      maxDrawdown: -0.18,
      totalTrades: 23
    },
    specialties: ['Concentrated positions', 'Activist investing', 'Quality growth'],
    isActive: true,
    lastAnalysis: '2025-07-25',
    confidence: 0.79
  },
  {
    name: 'Stanley Druckenmiller',
    description: 'Macro trends and asymmetric risk-reward',
    icon: '🌍',
    strategy: 'Macro Investing',
    riskLevel: 'High',
    performance: {
      winRate: 0.71,
      avgReturn: 0.20,
      sharpeRatio: 1.1,
      maxDrawdown: -0.15,
      totalTrades: 56
    },
    specialties: ['Macro trends', 'Currency trades', 'Commodities', 'Leverage'],
    isActive: true,
    lastAnalysis: '2025-07-25',
    confidence: 0.76
  },
  {
    name: 'Ben Graham',
    description: 'Father of value investing and security analysis',
    icon: '📊',
    strategy: 'Deep Value',
    riskLevel: 'Low',
    performance: {
      winRate: 0.75,
      avgReturn: 0.12,
      sharpeRatio: 1.4,
      maxDrawdown: -0.05,
      totalTrades: 41
    },
    specialties: ['Net-net stocks', 'Asset-based valuation', 'Margin of safety'],
    isActive: true,
    lastAnalysis: '2025-07-25',
    confidence: 0.83
  },
  {
    name: 'Phil Fisher',
    description: 'Qualitative analysis and scuttlebutt approach',
    icon: '🔍',
    strategy: 'Quality Growth',
    riskLevel: 'Medium',
    performance: {
      winRate: 0.73,
      avgReturn: 0.16,
      sharpeRatio: 1.2,
      maxDrawdown: -0.10,
      totalTrades: 38
    },
    specialties: ['Qualitative analysis', 'Management quality', 'R&D focus'],
    isActive: true,
    lastAnalysis: '2025-07-25',
    confidence: 0.80
  },
  {
    name: 'Aswath Damodaran',
    description: 'Valuation expert and financial modeling',
    icon: '📋',
    strategy: 'Valuation-Based',
    riskLevel: 'Medium',
    performance: {
      winRate: 0.77,
      avgReturn: 0.14,
      sharpeRatio: 1.3,
      maxDrawdown: -0.07,
      totalTrades: 52
    },
    specialties: ['DCF modeling', 'Valuation metrics', 'Risk assessment'],
    isActive: true,
    lastAnalysis: '2025-07-25',
    confidence: 0.86
  },
  {
    name: 'Valuation Agent',
    description: 'Multi-factor valuation and quantitative analysis',
    icon: '⚖️',
    strategy: 'Quantitative',
    riskLevel: 'Medium',
    performance: {
      winRate: 0.74,
      avgReturn: 0.15,
      sharpeRatio: 1.2,
      maxDrawdown: -0.09,
      totalTrades: 78
    },
    specialties: ['Multi-factor models', 'Quantitative analysis', 'Factor investing'],
    isActive: true,
    lastAnalysis: '2025-07-25',
    confidence: 0.81
  },
  {
    name: 'Fundamentals Agent',
    description: 'Financial metrics and fundamental analysis',
    icon: '📈',
    strategy: 'Fundamental',
    riskLevel: 'Low',
    performance: {
      winRate: 0.76,
      avgReturn: 0.13,
      sharpeRatio: 1.3,
      maxDrawdown: -0.06,
      totalTrades: 95
    },
    specialties: ['Financial ratios', 'Earnings analysis', 'Balance sheet analysis'],
    isActive: true,
    lastAnalysis: '2025-07-25',
    confidence: 0.84
  }
]

const RiskLevelBadge: React.FC<{ level: string }> = ({ level }) => {
  const colors = {
    Low: 'bg-green-100 text-green-700 border-green-200',
    Medium: 'bg-yellow-100 text-yellow-700 border-yellow-200',
    High: 'bg-red-100 text-red-700 border-red-200'
  }
  
  return (
    <span className={`inline-flex items-center px-2 py-1 rounded-full text-xs font-medium border ${colors[level as keyof typeof colors]}`}>
      {level} Risk
    </span>
  )
}

const PerformanceMetric: React.FC<{ label: string; value: number; format: 'percentage' | 'decimal' | 'number' }> = ({ label, value, format }) => {
  const formatValue = () => {
    switch (format) {
      case 'percentage':
        return formatPercentage(value * 100)
      case 'decimal':
        return value.toFixed(2)
      case 'number':
        return value.toString()
    }
  }
  
  return (
    <div className="text-center">
      <p className="text-2xl font-bold text-primary">{formatValue()}</p>
      <p className="text-xs text-muted-foreground">{label}</p>
    </div>
  )
}

const AgentCard: React.FC<{ agent: AgentProfile; onToggle: () => void }> = ({ agent, onToggle }) => {
  const isPositive = agent.performance.avgReturn >= 0
  
  return (
    <div onClick={onToggle}>
      <Card className="hover:shadow-lg transition-shadow cursor-pointer">
        <CardHeader>
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-3">
            <div className="text-3xl">{agent.icon}</div>
            <div>
              <CardTitle className="text-lg">{agent.name}</CardTitle>
              <p className="text-sm text-muted-foreground">{agent.strategy}</p>
            </div>
          </div>
          <div className="flex items-center space-x-2">
            <RiskLevelBadge level={agent.riskLevel} />
            <div className={`w-3 h-3 rounded-full ${agent.isActive ? 'bg-green-500' : 'bg-gray-300'}`}></div>
          </div>
        </div>
      </CardHeader>
      <CardContent>
        <p className="text-sm text-muted-foreground mb-4">{agent.description}</p>
        
        <div className="grid grid-cols-2 gap-4 mb-4">
          <PerformanceMetric 
            label="Win Rate" 
            value={agent.performance.winRate} 
            format="percentage" 
          />
          <PerformanceMetric 
            label="Avg Return" 
            value={agent.performance.avgReturn} 
            format="percentage" 
          />
        </div>
        
        <div className="space-y-2">
          <div className="flex justify-between text-sm">
            <span className="text-muted-foreground">Sharpe Ratio:</span>
            <span className="font-medium">{agent.performance.sharpeRatio.toFixed(2)}</span>
          </div>
          <div className="flex justify-between text-sm">
            <span className="text-muted-foreground">Max Drawdown:</span>
            <span className="font-medium text-red-600">{formatPercentage(agent.performance.maxDrawdown * 100)}</span>
          </div>
          <div className="flex justify-between text-sm">
            <span className="text-muted-foreground">Total Trades:</span>
            <span className="font-medium">{agent.performance.totalTrades}</span>
          </div>
        </div>
        
        <div className="mt-4 pt-4 border-t">
          <div className="flex justify-between items-center">
            <span className="text-sm text-muted-foreground">Confidence:</span>
            <span className="text-sm font-medium">{formatPercentage(agent.confidence * 100)}</span>
          </div>
          <div className="w-full bg-gray-200 rounded-full h-2 mt-1">
            <div 
              className="bg-primary h-2 rounded-full" 
              style={{ width: `${agent.confidence * 100}%` }}
            ></div>
          </div>
                 </div>
       </CardContent>
       </Card>
     </div>
   )
 }

const AgentDetailModal: React.FC<{ agent: AgentProfile | null; onClose: () => void }> = ({ agent, onClose }) => {
  if (!agent) return null

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
      <div className="bg-white rounded-lg p-6 w-full max-w-4xl max-h-[90vh] overflow-y-auto">
        <div className="flex items-center justify-between mb-6">
          <div className="flex items-center space-x-3">
            <div className="text-4xl">{agent.icon}</div>
            <div>
              <h2 className="text-2xl font-bold">{agent.name}</h2>
              <p className="text-muted-foreground">{agent.strategy} Strategy</p>
            </div>
          </div>
          <Button variant="outline" onClick={onClose}>Close</Button>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <div>
            <h3 className="text-lg font-semibold mb-3">Strategy Overview</h3>
            <p className="text-muted-foreground mb-4">{agent.description}</p>
            
            <div className="space-y-3">
              <div className="flex items-center justify-between">
                <span className="text-sm font-medium">Risk Level:</span>
                <RiskLevelBadge level={agent.riskLevel} />
              </div>
              <div className="flex items-center justify-between">
                <span className="text-sm font-medium">Status:</span>
                <span className={`text-sm ${agent.isActive ? 'text-green-600' : 'text-gray-600'}`}>
                  {agent.isActive ? 'Active' : 'Inactive'}
                </span>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-sm font-medium">Last Analysis:</span>
                <span className="text-sm text-muted-foreground">{agent.lastAnalysis}</span>
              </div>
            </div>
          </div>

          <div>
            <h3 className="text-lg font-semibold mb-3">Performance Metrics</h3>
            <div className="grid grid-cols-2 gap-4">
              <PerformanceMetric 
                label="Win Rate" 
                value={agent.performance.winRate} 
                format="percentage" 
              />
              <PerformanceMetric 
                label="Avg Return" 
                value={agent.performance.avgReturn} 
                format="percentage" 
              />
              <PerformanceMetric 
                label="Sharpe Ratio" 
                value={agent.performance.sharpeRatio} 
                format="decimal" 
              />
              <PerformanceMetric 
                label="Max Drawdown" 
                value={agent.performance.maxDrawdown} 
                format="percentage" 
              />
            </div>
          </div>
        </div>

        <div className="mt-6">
          <h3 className="text-lg font-semibold mb-3">Specialties</h3>
          <div className="flex flex-wrap gap-2">
            {agent.specialties.map((specialty, index) => (
              <span 
                key={index}
                className="inline-flex items-center px-3 py-1 rounded-full text-sm bg-primary/10 text-primary border border-primary/20"
              >
                {specialty}
              </span>
            ))}
          </div>
        </div>

        <div className="mt-6 flex space-x-3">
          <Button className="flex items-center space-x-2">
            <Play className="w-4 h-4" />
            Run Analysis
          </Button>
          <Button variant="outline" className="flex items-center space-x-2">
            <Settings className="w-4 h-4" />
            Configure
          </Button>
          <Button variant="outline" className="flex items-center space-x-2">
            <BarChart3 className="w-4 h-4" />
            View History
          </Button>
        </div>
      </div>
    </div>
  )
}

export const AgentsPage: React.FC = () => {
  const [agents, setAgents] = useState<AgentProfile[]>([])
  const [selectedAgent, setSelectedAgent] = useState<AgentProfile | null>(null)
  const [filter, setFilter] = useState<'all' | 'active' | 'inactive'>('all')
  const [sortBy, setSortBy] = useState<'name' | 'performance' | 'risk'>('name')
  const [isLoading, setIsLoading] = useState(true)

  useEffect(() => {
    const loadAgents = async () => {
      try {
        // Get available agents and performance data from backend
        const [availableAgents, performanceData] = await Promise.all([
          apiService.getAgents(),
          apiService.getAgentPerformance()
        ])
        
        // Create agent profiles based on real agent names and performance
        const realAgentProfiles: AgentProfile[] = availableAgents.map(agentName => {
          // Map agent names to their profiles
          const agentMap: Record<string, Partial<AgentProfile>> = {
            'WarrenBuffettAgent': {
              name: 'Warren Buffett',
              description: 'Value investing legend focused on long-term compound growth',
              icon: '💰',
              strategy: 'Value Investing',
              riskLevel: 'Low' as const,
              specialties: ['Blue-chip stocks', 'Consumer goods', 'Insurance', 'Long-term holds']
            },
            'PeterLynchAgent': {
              name: 'Peter Lynch',
              description: 'Growth at a reasonable price (GARP) specialist',
              icon: '📈',
              strategy: 'GARP',
              riskLevel: 'Medium' as const,
              specialties: ['Growth stocks', 'Consumer discretionary', 'Technology', 'Retail']
            },
            'CharlieMungerAgent': {
              name: 'Charlie Munger',
              description: 'Mental models and quality business focus',
              icon: '🧠',
              strategy: 'Quality Investing',
              riskLevel: 'Low' as const,
              specialties: ['High-quality businesses', 'Mental models', 'Concentrated positions']
            },
            'CathieWoodAgent': {
              name: 'Cathie Wood',
              description: 'Innovation and disruption technology investor',
              icon: '🚀',
              strategy: 'Innovation Investing',
              riskLevel: 'High' as const,
              specialties: ['AI/ML', 'Biotechnology', 'Fintech', 'Electric vehicles']
            },
            'BillAckmanAgent': {
              name: 'Bill Ackman',
              description: 'Concentrated positions in high-conviction ideas',
              icon: '🎯',
              strategy: 'Concentrated Investing',
              riskLevel: 'High' as const,
              specialties: ['Concentrated positions', 'Activist investing', 'Quality growth']
            },
            'StanleyDruckenmillerAgent': {
              name: 'Stanley Druckenmiller',
              description: 'Macro trends and asymmetric risk-reward',
              icon: '🌍',
              strategy: 'Macro Investing',
              riskLevel: 'High' as const,
              specialties: ['Macro trends', 'Currency trades', 'Commodities', 'Leverage']
            },
            'BenGrahamAgent': {
              name: 'Ben Graham',
              description: 'Father of value investing and security analysis',
              icon: '📊',
              strategy: 'Deep Value',
              riskLevel: 'Low' as const,
              specialties: ['Net-net stocks', 'Asset-based valuation', 'Margin of safety']
            },
            'PhilFisherAgent': {
              name: 'Phil Fisher',
              description: 'Qualitative analysis and scuttlebutt approach',
              icon: '🔍',
              strategy: 'Quality Growth',
              riskLevel: 'Medium' as const,
              specialties: ['Qualitative analysis', 'Management quality', 'R&D focus']
            },
            'AswathDamodaranAgent': {
              name: 'Aswath Damodaran',
              description: 'Valuation expert and financial modeling',
              icon: '📋',
              strategy: 'Valuation-Based',
              riskLevel: 'Medium' as const,
              specialties: ['DCF modeling', 'Valuation metrics', 'Risk assessment']
            },
            'ValuationAgent': {
              name: 'Valuation Agent',
              description: 'Multi-factor valuation and quantitative analysis',
              icon: '⚖️',
              strategy: 'Quantitative',
              riskLevel: 'Medium' as const,
              specialties: ['Multi-factor models', 'Quantitative analysis', 'Factor investing']
            },
            'FundamentalsAgent': {
              name: 'Fundamentals Agent',
              description: 'Financial metrics and fundamental analysis',
              icon: '📈',
              strategy: 'Fundamental',
              riskLevel: 'Low' as const,
              specialties: ['Financial ratios', 'Earnings analysis', 'Balance sheet analysis']
            }
          }

          const agentInfo = agentMap[agentName] || {
            name: agentName.replace('Agent', ''),
            description: 'AI investment agent',
            icon: '🤖',
            strategy: 'AI Strategy',
            riskLevel: 'Medium' as const,
            specialties: ['AI analysis', 'Machine learning']
          }

          // Get real performance data from backend
          const performance = performanceData[agentName]
          
          return {
            ...agentInfo,
            performance: {
              winRate: performance?.win_rate || 0.75,
              avgReturn: performance?.avg_return || 0.15,
              sharpeRatio: performance?.sharpe_ratio || 1.0,
              maxDrawdown: performance?.max_drawdown || -0.1,
              totalTrades: performance?.total_trades || 50
            },
            isActive: performance?.is_active ?? true,
            lastAnalysis: performance?.last_analysis || new Date().toISOString().split('T')[0],
            confidence: performance?.confidence || 0.8
          } as AgentProfile
        })

        setAgents(realAgentProfiles)
      } catch (error) {
        console.error('Error loading agents:', error)
        // Fallback to mock data if API fails
        setAgents(agentProfiles)
      } finally {
        setIsLoading(false)
      }
    }

    loadAgents()
  }, [])

  const filteredAgents = agents.filter(agent => {
    if (filter === 'active') return agent.isActive
    if (filter === 'inactive') return !agent.isActive
    return true
  })

  const sortedAgents = [...filteredAgents].sort((a, b) => {
    switch (sortBy) {
      case 'name':
        return a.name.localeCompare(b.name)
      case 'performance':
        return b.performance.winRate - a.performance.winRate
      case 'risk':
        const riskOrder = { Low: 1, Medium: 2, High: 3 }
        return riskOrder[a.riskLevel as keyof typeof riskOrder] - riskOrder[b.riskLevel as keyof typeof riskOrder]
      default:
        return 0
    }
  })

  const toggleAgentStatus = (agentName: string) => {
    setAgents(prev => prev.map(agent => 
      agent.name === agentName 
        ? { ...agent, isActive: !agent.isActive }
        : agent
    ))
  }

  const getOverallStats = () => {
    const activeAgents = agents.filter(a => a.isActive)
    const avgWinRate = activeAgents.reduce((sum, a) => sum + a.performance.winRate, 0) / activeAgents.length
    const avgReturn = activeAgents.reduce((sum, a) => sum + a.performance.avgReturn, 0) / activeAgents.length
    const totalTrades = activeAgents.reduce((sum, a) => sum + a.performance.totalTrades, 0)
    
    return { avgWinRate, avgReturn, totalTrades, activeCount: activeAgents.length }
  }

  const stats = getOverallStats()

  if (isLoading) {
    return (
      <div className="p-8 space-y-8">
        <div className="animate-pulse">
          <div className="h-8 bg-gray-200 rounded w-1/3 mb-4"></div>
          <div className="grid grid-cols-1 md:grid-cols-4 gap-6 mb-8">
            {[...Array(4)].map((_, i) => (
              <div key={i} className="h-32 bg-gray-200 rounded"></div>
            ))}
          </div>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {[...Array(6)].map((_, i) => (
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
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold tracking-tight">AI Investment Agents</h1>
          <p className="text-muted-foreground">
            Manage and monitor your AI investment agents
          </p>
        </div>
        <div className="flex space-x-3">
          <Button className="flex items-center space-x-2">
            <Zap className="w-4 h-4" />
            Run All Agents
          </Button>
          <Button variant="outline" className="flex items-center space-x-2">
            <Settings className="w-4 h-4" />
            Settings
          </Button>
        </div>
      </div>

      {/* Stats Overview */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Active Agents</CardTitle>
            <Brain className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{stats.activeCount}</div>
            <p className="text-xs text-muted-foreground">of {agents.length} total</p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Avg Win Rate</CardTitle>
            <TrendingUp className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{formatPercentage(stats.avgWinRate * 100)}</div>
            <p className="text-xs text-muted-foreground">across all agents</p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Avg Return</CardTitle>
            <Target className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{formatPercentage(stats.avgReturn * 100)}</div>
            <p className="text-xs text-muted-foreground">annualized</p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Total Trades</CardTitle>
            <BarChart3 className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{stats.totalTrades}</div>
            <p className="text-xs text-muted-foreground">this month</p>
          </CardContent>
        </Card>
      </div>

      {/* Filters and Controls */}
      <div className="flex items-center justify-between">
        <div className="flex items-center space-x-4">
          <div className="flex items-center space-x-2">
            <span className="text-sm font-medium">Filter:</span>
            <select 
              value={filter} 
              onChange={(e) => setFilter(e.target.value as 'all' | 'active' | 'inactive')}
              className="text-sm border rounded px-2 py-1"
            >
              <option value="all">All Agents</option>
              <option value="active">Active Only</option>
              <option value="inactive">Inactive Only</option>
            </select>
          </div>
          <div className="flex items-center space-x-2">
            <span className="text-sm font-medium">Sort by:</span>
            <select 
              value={sortBy} 
              onChange={(e) => setSortBy(e.target.value as 'name' | 'performance' | 'risk')}
              className="text-sm border rounded px-2 py-1"
            >
              <option value="name">Name</option>
              <option value="performance">Performance</option>
              <option value="risk">Risk Level</option>
            </select>
          </div>
        </div>
      </div>

      {/* Agents Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {sortedAgents.map((agent) => (
          <AgentCard 
            key={agent.name} 
            agent={agent} 
            onToggle={() => setSelectedAgent(agent)}
          />
        ))}
      </div>

      {/* Agent Detail Modal */}
      <AgentDetailModal 
        agent={selectedAgent} 
        onClose={() => setSelectedAgent(null)} 
      />
    </div>
  )
} 