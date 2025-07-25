import React, { useState } from 'react'
import { Search, Filter, Download, Play } from 'lucide-react'
import { Card, CardContent, CardHeader, CardTitle } from '../components/ui/Card'
import { Button } from '../components/ui/Button'
import { Input } from '../components/ui/Input'
import { formatCurrency, formatPercentage, getSignalColor, getScoreColor } from '../lib/utils'
import { apiService } from '../services/api'
import type { Agent, StockAnalysis as StockAnalysisType, AgentAnalysis } from '../types'

const availableAgents: Agent[] = [
  { name: 'WarrenBuffettAgent', description: 'Value investing principles', icon: '💰', isActive: true },
  { name: 'PeterLynchAgent', description: 'Growth at a reasonable price', icon: '📈', isActive: true },
  { name: 'CharlieMungerAgent', description: 'Mental models approach', icon: '🧠', isActive: true },
  { name: 'CathieWoodAgent', description: 'Innovation and disruption', icon: '🚀', isActive: true },
  { name: 'BillAckmanAgent', description: 'Concentrated positions', icon: '🎯', isActive: true },
  { name: 'StanleyDruckenmillerAgent', description: 'Macro trends', icon: '🌍', isActive: true },
  { name: 'BenGrahamAgent', description: 'Security analysis', icon: '📊', isActive: true },
  { name: 'PhilFisherAgent', description: 'Qualitative analysis', icon: '🔍', isActive: true },
  { name: 'AswathDamodaranAgent', description: 'Valuation expert', icon: '📋', isActive: true },
  { name: 'ValuationAgent', description: 'Multi-factor valuation', icon: '⚖️', isActive: true },
  { name: 'FundamentalsAgent', description: 'Financial metrics', icon: '📈', isActive: true },
]



const AgentCard: React.FC<{
  agent: Agent
  isSelected: boolean
  onToggle: () => void
}> = ({ agent, isSelected, onToggle }) => (
  <div
    className={`p-4 border rounded-lg cursor-pointer transition-colors ${
      isSelected ? 'border-primary bg-primary/5' : 'border-border hover:border-primary/50'
    }`}
    onClick={onToggle}
  >
    <div className="flex items-center space-x-3">
      <div className="text-2xl">{agent.icon}</div>
      <div className="flex-1">
        <h3 className="font-medium text-sm">{agent.name.replace('Agent', '')}</h3>
        <p className="text-xs text-muted-foreground">{agent.description}</p>
      </div>
      <div className={`w-4 h-4 rounded border-2 ${
        isSelected ? 'bg-primary border-primary' : 'border-muted-foreground'
      }`}>
        {isSelected && (
          <div className="w-full h-full bg-primary rounded-sm flex items-center justify-center">
            <div className="w-2 h-2 bg-white rounded-sm"></div>
          </div>
        )}
      </div>
    </div>
  </div>
)

const AnalysisResultCard: React.FC<{
  analysis: StockAnalysisType
}> = ({ analysis }) => (
  <Card>
    <CardHeader>
      <div className="flex items-center justify-between">
        <div className="flex items-center space-x-3">
          <div className="w-10 h-10 bg-primary/10 rounded-lg flex items-center justify-center">
            <span className="text-sm font-semibold text-primary">{analysis.ticker}</span>
          </div>
          <div>
            <CardTitle className="text-lg">{analysis.company_name}</CardTitle>
            <p className="text-sm text-muted-foreground">{analysis.ticker}</p>
          </div>
        </div>
        <div className="text-right">
          <div className={`inline-flex items-center px-3 py-1 rounded-full text-sm font-medium border ${getSignalColor(analysis.signal)}`}>
            {analysis.signal.toUpperCase()}
          </div>
          <p className={`text-lg font-bold mt-1 ${getScoreColor(analysis.score)}`}>
            {analysis.score}/10
          </p>
        </div>
      </div>
    </CardHeader>
    <CardContent>
      <div className="grid grid-cols-3 gap-4 mb-4">
        <div className="text-center">
          <p className="text-2xl font-bold text-green-600">{analysis.bullish}</p>
          <p className="text-xs text-muted-foreground">Bullish</p>
        </div>
        <div className="text-center">
          <p className="text-2xl font-bold text-red-600">{analysis.bearish}</p>
          <p className="text-xs text-muted-foreground">Bearish</p>
        </div>
        <div className="text-center">
          <p className="text-2xl font-bold text-yellow-600">{analysis.neutral}</p>
          <p className="text-xs text-muted-foreground">Neutral</p>
        </div>
      </div>
      
      {/* Agent Details */}
      <div className="space-y-3">
        <h4 className="font-medium text-sm">Agent Analysis Details</h4>
        {Object.entries(analysis.agent_analyses).map(([agentName, agentAnalysis]) => (
          <div key={agentName} className="p-3 bg-muted/50 rounded-lg">
            <div className="flex items-center justify-between mb-2">
              <span className="font-medium text-sm">{agentAnalysis.name}</span>
              <div className={`inline-flex items-center px-2 py-1 rounded-full text-xs font-medium ${getSignalColor(agentAnalysis.signal)}`}>
                {agentAnalysis.signal.toUpperCase()}
              </div>
            </div>
            <p className="text-xs text-muted-foreground mb-2">{agentAnalysis.reasoning}</p>
            <div className="flex items-center justify-between text-xs">
              <span>Score: <span className={getScoreColor(agentAnalysis.score)}>{agentAnalysis.score}/10</span></span>
              <span>Confidence: {formatPercentage(agentAnalysis.confidence * 100)}</span>
            </div>
          </div>
        ))}
      </div>
    </CardContent>
  </Card>
)

export const StockAnalysis: React.FC = () => {
  const [tickers, setTickers] = useState('')
  const [selectedAgents, setSelectedAgents] = useState<string[]>(['WarrenBuffettAgent', 'PeterLynchAgent'])
  const [period, setPeriod] = useState<'Annual' | 'Quarterly'>('Annual')
  const [isAnalyzing, setIsAnalyzing] = useState(false)
  const [results, setResults] = useState<StockAnalysisType[]>([])

  const handleAgentToggle = (agentName: string) => {
    setSelectedAgents(prev => 
      prev.includes(agentName) 
        ? prev.filter(name => name !== agentName)
        : [...prev, agentName]
    )
  }

  const handleAnalyze = async () => {
    if (!tickers.trim()) return
    
    setIsAnalyzing(true)
    try {
      const tickerList = tickers.split(',').map(t => t.trim().toUpperCase()).filter(t => t);
      const request = {
        tickers: tickerList,
        agents: selectedAgents,
        period: period,
        include_technical: true,
        include_fundamentals: true,
        include_predictions: false
      };
      
      const analysisResponse = await apiService.analyzeStocks(request);
      setResults(analysisResponse.summary);
    } catch (error) {
      console.error('Error analyzing stocks:', error);
      // Optionally show an error message to the user
    } finally {
      setIsAnalyzing(false);
    }
  }

  return (
    <div className="p-8 space-y-8">
      {/* Header */}
      <div>
        <h1 className="text-3xl font-bold tracking-tight">Stock Analysis</h1>
        <p className="text-muted-foreground">
          Analyze stocks using multiple AI agents with different investment philosophies.
        </p>
      </div>

      {/* Analysis Form */}
      <Card>
        <CardHeader>
          <CardTitle>Analysis Configuration</CardTitle>
        </CardHeader>
        <CardContent className="space-y-6">
          {/* Ticker Input */}
          <div>
            <Input
              label="Stock Tickers"
              placeholder="AAPL, MSFT, GOOGL (comma separated)"
              value={tickers}
              onChange={(e) => setTickers(e.target.value)}
            />
            <p className="text-xs text-muted-foreground mt-1">
              Enter stock tickers separated by commas
            </p>
          </div>

          {/* Period Selection */}
          <div>
            <label className="text-sm font-medium">Analysis Period</label>
            <div className="flex space-x-4 mt-2">
              <label className="flex items-center space-x-2">
                <input
                  type="radio"
                  value="Annual"
                  checked={period === 'Annual'}
                  onChange={(e) => setPeriod(e.target.value as 'Annual' | 'Quarterly')}
                  className="text-primary"
                />
                <span className="text-sm">Annual</span>
              </label>
              <label className="flex items-center space-x-2">
                <input
                  type="radio"
                  value="Quarterly"
                  checked={period === 'Quarterly'}
                  onChange={(e) => setPeriod(e.target.value as 'Annual' | 'Quarterly')}
                  className="text-primary"
                />
                <span className="text-sm">Quarterly</span>
              </label>
            </div>
          </div>

          {/* Agent Selection */}
          <div>
            <label className="text-sm font-medium">Select Agents</label>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3 mt-2">
              {availableAgents.map((agent) => (
                <AgentCard
                  key={agent.name}
                  agent={agent}
                  isSelected={selectedAgents.includes(agent.name)}
                  onToggle={() => handleAgentToggle(agent.name)}
                />
              ))}
            </div>
          </div>

          {/* Action Buttons */}
          <div className="flex space-x-3">
            <Button
              onClick={handleAnalyze}
              disabled={isAnalyzing || !tickers.trim()}
              className="flex items-center space-x-2"
            >
              <Play className="w-4 h-4" />
              {isAnalyzing ? 'Analyzing...' : 'Run Analysis'}
            </Button>
            <Button variant="outline" className="flex items-center space-x-2">
              <Download className="w-4 h-4" />
              Export Results
            </Button>
          </div>
        </CardContent>
      </Card>

      {/* Results */}
      {results.length > 0 && (
        <div className="space-y-6">
          <div className="flex items-center justify-between">
            <h2 className="text-2xl font-bold">Analysis Results</h2>
            <div className="flex items-center space-x-2">
              <Button variant="outline" size="sm">
                <Filter className="w-4 h-4 mr-2" />
                Filter
              </Button>
              <Button variant="outline" size="sm">
                <Download className="w-4 h-4 mr-2" />
                Export
              </Button>
            </div>
          </div>

          <div className="grid gap-6">
            {results.map((analysis) => (
              <AnalysisResultCard key={analysis.ticker} analysis={analysis} />
            ))}
          </div>
        </div>
      )}
    </div>
  )
} 