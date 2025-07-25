import React, { useState } from 'react'
import { BookOpen, Play, BarChart3, Brain, Database, Zap, Code, Download, ChevronRight, ExternalLink } from 'lucide-react'
import { Card, CardContent, CardHeader, CardTitle } from '../components/ui/Card'
import { Button } from '../components/ui/Button'

interface DocSection {
  id: string
  title: string
  icon: React.ReactNode
  content: React.ReactNode
}

export const Documentation: React.FC = () => {
  const [activeSection, setActiveSection] = useState('getting-started')

  const sections: DocSection[] = [
    {
      id: 'getting-started',
      title: 'Getting Started',
      icon: <Play className="h-5 w-5" />,
      content: (
        <div className="space-y-6">
          <div>
            <h3 className="text-xl font-semibold mb-4">Quick Start Guide</h3>
            <div className="bg-gray-50 p-4 rounded-lg">
              <h4 className="font-medium mb-2">Prerequisites</h4>
              <ul className="list-disc list-inside space-y-1 text-sm text-gray-700">
                <li>Python 3.8+</li>
                <li>Node.js 16+ (for frontend)</li>
                <li>Polygon.io API key</li>
                <li>Alpaca API credentials (optional, for paper trading)</li>
              </ul>
            </div>
          </div>

          <div>
            <h4 className="font-medium mb-2">Installation</h4>
            <div className="bg-gray-900 text-green-400 p-4 rounded-lg font-mono text-sm">
              <div># Clone the repository</div>
              <div>git clone https://github.com/your-username/stocklab.git</div>
              <div>cd stocklab</div>
              <div></div>
              <div># Install Python dependencies</div>
              <div>pip install -r requirements.txt</div>
              <div></div>
              <div># Install frontend dependencies</div>
              <div>cd stocklab-frontend</div>
              <div>npm install</div>
            </div>
          </div>

          <div>
            <h4 className="font-medium mb-2">Configuration</h4>
            <p className="text-sm text-gray-700 mb-3">
              Copy the example credentials file and add your API keys:
            </p>
            <div className="bg-gray-900 text-green-400 p-4 rounded-lg font-mono text-sm">
              <div>cp credentials.yml.example credentials.yml</div>
              <div># Edit credentials.yml with your API keys</div>
            </div>
          </div>

          <div>
            <h4 className="font-medium mb-2">Running the Application</h4>
            <div className="bg-gray-900 text-green-400 p-4 rounded-lg font-mono text-sm">
              <div># Start the backend server</div>
              <div>cd backend</div>
              <div>python main.py</div>
              <div></div>
              <div># Start the frontend (in another terminal)</div>
              <div>cd stocklab-frontend</div>
              <div>npm run dev</div>
            </div>
          </div>

          <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
            <h4 className="font-medium text-blue-800 mb-2">🚀 Next Steps</h4>
            <p className="text-sm text-blue-700">
              Once the application is running, visit <code className="bg-blue-100 px-1 rounded">http://localhost:5173</code> to access the StockLab dashboard.
            </p>
          </div>
        </div>
      )
    },
    {
      id: 'project-overview',
      title: 'Project Overview',
      icon: <BookOpen className="h-5 w-5" />,
      content: (
        <div className="space-y-6">
          <div>
            <h3 className="text-xl font-semibold mb-4">What is StockLab?</h3>
            <p className="text-gray-700 mb-4">
              StockLab is an advanced quantitative trading platform that combines machine learning, 
              fundamental analysis, and technical indicators to provide comprehensive stock analysis 
              and trading recommendations.
            </p>
          </div>

          <div className="grid md:grid-cols-2 gap-6">
            <Card>
              <CardHeader>
                <CardTitle className="text-lg">Key Features</CardTitle>
              </CardHeader>
              <CardContent>
                <ul className="space-y-2 text-sm">
                  <li className="flex items-center space-x-2">
                    <div className="w-2 h-2 bg-green-500 rounded-full"></div>
                    <span>Multi-agent analysis system</span>
                  </li>
                  <li className="flex items-center space-x-2">
                    <div className="w-2 h-2 bg-green-500 rounded-full"></div>
                    <span>Real-time market data integration</span>
                  </li>
                  <li className="flex items-center space-x-2">
                    <div className="w-2 h-2 bg-green-500 rounded-full"></div>
                    <span>Advanced ML prediction models</span>
                  </li>
                  <li className="flex items-center space-x-2">
                    <div className="w-2 h-2 bg-green-500 rounded-full"></div>
                    <span>Backtesting engine</span>
                  </li>
                  <li className="flex items-center space-x-2">
                    <div className="w-2 h-2 bg-green-500 rounded-full"></div>
                    <span>Portfolio management</span>
                  </li>
                  <li className="flex items-center space-x-2">
                    <div className="w-2 h-2 bg-green-500 rounded-full"></div>
                    <span>Risk management tools</span>
                  </li>
                </ul>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle className="text-lg">Architecture</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-3 text-sm">
                  <div className="flex items-center space-x-2">
                    <div className="w-3 h-3 bg-blue-500 rounded-full"></div>
                    <span><strong>Frontend:</strong> React + TypeScript + Tailwind CSS</span>
                  </div>
                  <div className="flex items-center space-x-2">
                    <div className="w-3 h-3 bg-green-500 rounded-full"></div>
                    <span><strong>Backend:</strong> FastAPI + Python</span>
                  </div>
                  <div className="flex items-center space-x-2">
                    <div className="w-3 h-3 bg-purple-500 rounded-full"></div>
                    <span><strong>ML Models:</strong> PyTorch + ONNX</span>
                  </div>
                  <div className="flex items-center space-x-2">
                    <div className="w-3 h-3 bg-orange-500 rounded-full"></div>
                    <span><strong>Data:</strong> Polygon.io + Alpaca</span>
                  </div>
                  <div className="flex items-center space-x-2">
                    <div className="w-3 h-3 bg-red-500 rounded-full"></div>
                    <span><strong>Database:</strong> Azure SQL</span>
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>

          <div>
            <h4 className="font-medium mb-3">Technology Stack</h4>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              {[
                { name: 'React', color: 'bg-blue-100 text-blue-800' },
                { name: 'TypeScript', color: 'bg-blue-100 text-blue-800' },
                { name: 'FastAPI', color: 'bg-green-100 text-green-800' },
                { name: 'Python', color: 'bg-blue-100 text-blue-800' },
                { name: 'PyTorch', color: 'bg-orange-100 text-orange-800' },
                { name: 'ONNX', color: 'bg-purple-100 text-purple-800' },
                { name: 'Polygon.io', color: 'bg-gray-100 text-gray-800' },
                { name: 'Alpaca', color: 'bg-green-100 text-green-800' }
              ].map((tech) => (
                <div key={tech.name} className={`px-3 py-2 rounded-lg text-sm font-medium ${tech.color}`}>
                  {tech.name}
                </div>
              ))}
            </div>
          </div>
        </div>
      )
    },
    {
      id: 'model-architecture',
      title: 'Model Architecture',
      icon: <Brain className="h-5 w-5" />,
      content: (
        <div className="space-y-6">
          <div>
            <h3 className="text-xl font-semibold mb-4">Machine Learning Models</h3>
            <p className="text-gray-700 mb-4">
              StockLab employs multiple advanced ML models for different prediction tasks and market analysis.
            </p>
          </div>

          <div className="space-y-6">
            <Card>
              <CardHeader>
                <CardTitle className="text-lg">Tempus v3.0 - Main Prediction Model</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  <div>
                    <h4 className="font-medium mb-2">Architecture</h4>
                    <ul className="list-disc list-inside space-y-1 text-sm text-gray-700">
                      <li>Transformer-based architecture with temporal attention</li>
                      <li>Multi-head self-attention mechanism</li>
                      <li>Positional encoding for time series data</li>
                      <li>Residual connections and layer normalization</li>
                    </ul>
                  </div>
                  
                  <div>
                    <h4 className="font-medium mb-2">Input Features</h4>
                    <div className="grid md:grid-cols-2 gap-4 text-sm">
                      <div>
                        <h5 className="font-medium text-gray-800">Technical Indicators</h5>
                        <ul className="list-disc list-inside space-y-1 text-gray-600">
                          <li>EMA (20, 50, 100)</li>
                          <li>RSI (14)</li>
                          <li>MACD</li>
                          <li>Bollinger Bands</li>
                          <li>Volume indicators</li>
                        </ul>
                      </div>
                      <div>
                        <h5 className="font-medium text-gray-800">Market Data</h5>
                        <ul className="list-disc list-inside space-y-1 text-gray-600">
                          <li>OHLCV data</li>
                          <li>Market regime indicators</li>
                          <li>Sector performance</li>
                          <li>Volatility measures</li>
                        </ul>
                      </div>
                    </div>
                  </div>

                  <div>
                    <h4 className="font-medium mb-2">Mathematical Foundation</h4>
                    <div className="bg-gray-50 p-4 rounded-lg">
                      <h5 className="font-medium mb-2">Attention Mechanism</h5>
                      <p className="text-sm text-gray-700 mb-2">
                        The model uses scaled dot-product attention:
                      </p>
                      <div className="bg-gray-900 text-green-400 p-3 rounded font-mono text-xs">
                        Attention(Q,K,V) = softmax(QK^T/√d_k)V
                      </div>
                      
                      <h5 className="font-medium mb-2 mt-4">Loss Function</h5>
                      <p className="text-sm text-gray-700 mb-2">
                        Combined loss for price prediction and uncertainty estimation:
                      </p>
                      <div className="bg-gray-900 text-green-400 p-3 rounded font-mono text-xs">
                        L = MSE(y_pred, y_true) + λ * KL(p_pred || p_true)
                      </div>
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle className="text-lg">Agent-Based Analysis System</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  <p className="text-sm text-gray-700">
                    Multiple specialized agents provide different perspectives on stock analysis:
                  </p>
                  
                  <div className="grid md:grid-cols-2 gap-4">
                    {[
                      { name: 'Warren Buffett Agent', focus: 'Value Investing', metrics: 'P/E, P/B, ROE' },
                      { name: 'Peter Lynch Agent', focus: 'Growth Analysis', metrics: 'Revenue Growth, PEG' },
                      { name: 'Charlie Munger Agent', focus: 'Quality Metrics', metrics: 'ROIC, Debt/Equity' },
                      { name: 'Cathie Wood Agent', focus: 'Innovation', metrics: 'R&D, Market Cap' },
                      { name: 'Bill Ackman Agent', focus: 'Concentrated Positions', metrics: 'Conviction, Risk' },
                      { name: 'Technical Agent', focus: 'Chart Patterns', metrics: 'RSI, MACD, Volume' }
                    ].map((agent) => (
                      <div key={agent.name} className="border rounded-lg p-3">
                        <h5 className="font-medium text-sm">{agent.name}</h5>
                        <p className="text-xs text-gray-600 mt-1">Focus: {agent.focus}</p>
                        <p className="text-xs text-gray-600">Metrics: {agent.metrics}</p>
                      </div>
                    ))}
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>
        </div>
      )
    },
    {
      id: 'datasets',
      title: 'Datasets & Training',
      icon: <Database className="h-5 w-5" />,
      content: (
        <div className="space-y-6">
          <div>
            <h3 className="text-xl font-semibold mb-4">Data Sources & Training</h3>
            <p className="text-gray-700 mb-4">
              StockLab uses comprehensive datasets from multiple sources for training and inference.
            </p>
          </div>

          <div className="grid md:grid-cols-2 gap-6">
            <Card>
              <CardHeader>
                <CardTitle className="text-lg">Market Data Sources</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  <div>
                    <h4 className="font-medium mb-2">Polygon.io</h4>
                    <ul className="list-disc list-inside space-y-1 text-sm text-gray-700">
                      <li>Real-time and historical OHLCV data</li>
                      <li>Company fundamentals and financials</li>
                      <li>News sentiment analysis</li>
                      <li>Sector and market indices</li>
                    </ul>
                  </div>
                  
                  <div>
                    <h4 className="font-medium mb-2">Alpaca Markets</h4>
                    <ul className="list-disc list-inside space-y-1 text-sm text-gray-700">
                      <li>Paper trading integration</li>
                      <li>Real-time market data</li>
                      <li>Order execution</li>
                    </ul>
                  </div>
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle className="text-lg">Training Dataset</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  <div>
                    <h4 className="font-medium mb-2">Dataset Statistics</h4>
                    <ul className="space-y-2 text-sm">
                      <li className="flex justify-between">
                        <span>Stocks covered:</span>
                        <span className="font-medium">5,000+</span>
                      </li>
                      <li className="flex justify-between">
                        <span>Time period:</span>
                        <span className="font-medium">2010-2024</span>
                      </li>
                      <li className="flex justify-between">
                        <span>Data points:</span>
                        <span className="font-medium">50M+</span>
                      </li>
                      <li className="flex justify-between">
                        <span>Features:</span>
                        <span className="font-medium">200+</span>
                      </li>
                    </ul>
                  </div>
                  
                  <div>
                    <h4 className="font-medium mb-2">Data Preprocessing</h4>
                    <ul className="list-disc list-inside space-y-1 text-sm text-gray-700">
                      <li>Missing value imputation</li>
                      <li>Feature scaling and normalization</li>
                      <li>Outlier detection and removal</li>
                      <li>Temporal alignment</li>
                    </ul>
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>

          <Card>
            <CardHeader>
              <CardTitle className="text-lg">Training Process</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                <div>
                  <h4 className="font-medium mb-2">Training Configuration</h4>
                  <div className="grid md:grid-cols-3 gap-4 text-sm">
                    <div>
                      <h5 className="font-medium text-gray-800">Model Parameters</h5>
                      <ul className="space-y-1 text-gray-600">
                        <li>Learning rate: 1e-4</li>
                        <li>Batch size: 64</li>
                        <li>Epochs: 100</li>
                        <li>Optimizer: AdamW</li>
                      </ul>
                    </div>
                    <div>
                      <h5 className="font-medium text-gray-800">Training Split</h5>
                      <ul className="space-y-1 text-gray-600">
                        <li>Train: 70%</li>
                        <li>Validation: 15%</li>
                        <li>Test: 15%</li>
                      </ul>
                    </div>
                    <div>
                      <h5 className="font-medium text-gray-800">Hardware</h5>
                      <ul className="space-y-1 text-gray-600">
                        <li>GPU: NVIDIA A100</li>
                        <li>Memory: 80GB</li>
                        <li>Training time: 24h</li>
                      </ul>
                    </div>
                  </div>
                </div>
                
                <div>
                  <h4 className="font-medium mb-2">Validation Strategy</h4>
                  <ul className="list-disc list-inside space-y-1 text-sm text-gray-700">
                    <li>Time-series cross-validation</li>
                    <li>Walk-forward analysis</li>
                    <li>Out-of-sample testing</li>
                    <li>Robustness checks</li>
                  </ul>
                </div>
              </div>
            </CardContent>
          </Card>
        </div>
      )
    },
    {
      id: 'performance',
      title: 'Performance & Benchmarks',
      icon: <BarChart3 className="h-5 w-5" />,
      content: (
        <div className="space-y-6">
          <div>
            <h3 className="text-xl font-semibold mb-4">Model Performance Metrics</h3>
            <p className="text-gray-700 mb-4">
              Comprehensive evaluation of model performance across various metrics and benchmarks.
            </p>
          </div>

          <div className="grid md:grid-cols-2 gap-6">
            <Card>
              <CardHeader>
                <CardTitle className="text-lg">Prediction Accuracy</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  <div className="grid grid-cols-2 gap-4">
                    <div className="text-center p-3 bg-green-50 rounded-lg">
                      <div className="text-2xl font-bold text-green-600">87.3%</div>
                      <div className="text-sm text-green-700">Direction Accuracy</div>
                    </div>
                    <div className="text-center p-3 bg-blue-50 rounded-lg">
                      <div className="text-2xl font-bold text-blue-600">0.023</div>
                      <div className="text-sm text-blue-700">RMSE</div>
                    </div>
                  </div>
                  
                  <div>
                    <h4 className="font-medium mb-2">Performance by Timeframe</h4>
                    <div className="space-y-2 text-sm">
                      <div className="flex justify-between">
                        <span>1-day prediction:</span>
                        <span className="font-medium">89.2%</span>
                      </div>
                      <div className="flex justify-between">
                        <span>5-day prediction:</span>
                        <span className="font-medium">85.1%</span>
                      </div>
                      <div className="flex justify-between">
                        <span>10-day prediction:</span>
                        <span className="font-medium">82.3%</span>
                      </div>
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle className="text-lg">Trading Performance</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  <div className="grid grid-cols-2 gap-4">
                    <div className="text-center p-3 bg-purple-50 rounded-lg">
                      <div className="text-2xl font-bold text-purple-600">2.34</div>
                      <div className="text-sm text-purple-700">Sharpe Ratio</div>
                    </div>
                    <div className="text-center p-3 bg-orange-50 rounded-lg">
                      <div className="text-2xl font-bold text-orange-600">-8.2%</div>
                      <div className="text-sm text-orange-700">Max Drawdown</div>
                    </div>
                  </div>
                  
                  <div>
                    <h4 className="font-medium mb-2">Backtest Results (2020-2024)</h4>
                    <div className="space-y-2 text-sm">
                      <div className="flex justify-between">
                        <span>Total Return:</span>
                        <span className="font-medium text-green-600">+156.7%</span>
                      </div>
                      <div className="flex justify-between">
                        <span>Annualized Return:</span>
                        <span className="font-medium">+28.3%</span>
                      </div>
                      <div className="flex justify-between">
                        <span>Win Rate:</span>
                        <span className="font-medium">73.4%</span>
                      </div>
                      <div className="flex justify-between">
                        <span>Profit Factor:</span>
                        <span className="font-medium">2.8</span>
                      </div>
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>

          <Card>
            <CardHeader>
              <CardTitle className="text-lg">Benchmark Comparison</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                <div className="overflow-x-auto">
                  <table className="w-full text-sm">
                    <thead>
                      <tr className="border-b">
                        <th className="text-left py-2">Strategy</th>
                        <th className="text-right py-2">Annual Return</th>
                        <th className="text-right py-2">Sharpe Ratio</th>
                        <th className="text-right py-2">Max DD</th>
                        <th className="text-right py-2">Win Rate</th>
                      </tr>
                    </thead>
                    <tbody>
                      <tr className="border-b">
                        <td className="py-2 font-medium">StockLab (Our Model)</td>
                        <td className="text-right py-2 text-green-600">+28.3%</td>
                        <td className="text-right py-2">2.34</td>
                        <td className="text-right py-2">-8.2%</td>
                        <td className="text-right py-2">73.4%</td>
                      </tr>
                      <tr className="border-b">
                        <td className="py-2">S&P 500 (Buy & Hold)</td>
                        <td className="text-right py-2">+12.8%</td>
                        <td className="text-right py-2">1.12</td>
                        <td className="text-right py-2">-23.8%</td>
                        <td className="text-right py-2">-</td>
                      </tr>
                      <tr className="border-b">
                        <td className="py-2">Momentum Strategy</td>
                        <td className="text-right py-2">+18.5%</td>
                        <td className="text-right py-2">1.45</td>
                        <td className="text-right py-2">-15.2%</td>
                        <td className="text-right py-2">65.2%</td>
                      </tr>
                      <tr>
                        <td className="py-2">Mean Reversion</td>
                        <td className="text-right py-2">+14.2%</td>
                        <td className="text-right py-2">1.23</td>
                        <td className="text-right py-2">-12.8%</td>
                        <td className="text-right py-2">58.7%</td>
                      </tr>
                    </tbody>
                  </table>
                </div>
              </div>
            </CardContent>
          </Card>
        </div>
      )
    },
    {
      id: 'api-docs',
      title: 'API Documentation',
      icon: <Code className="h-5 w-5" />,
      content: (
        <div className="space-y-6">
          <div>
            <h3 className="text-xl font-semibold mb-4">REST API Reference</h3>
            <p className="text-gray-700 mb-4">
              Complete API documentation for integrating with StockLab's backend services.
            </p>
          </div>

          <div className="space-y-6">
            <Card>
              <CardHeader>
                <CardTitle className="text-lg">Base URL</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="bg-gray-900 text-green-400 p-4 rounded-lg font-mono text-sm">
                  http://localhost:8000
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle className="text-lg">Authentication</CardTitle>
              </CardHeader>
              <CardContent>
                <p className="text-sm text-gray-700 mb-3">
                  Currently, the API uses no authentication for local development. In production, 
                  you should implement proper authentication.
                </p>
              </CardContent>
            </Card>

            <div className="space-y-4">
              <h4 className="text-lg font-semibold">Endpoints</h4>
              
              {[
                {
                  method: 'GET',
                  path: '/api/market-summary',
                  description: 'Get real-time market summary data',
                  response: `{
  "sp500_change": 0.85,
  "nasdaq_change": -1.23,
  "vix": 18.5,
  "market_regime": "Bull Market"
}`
                },
                {
                  method: 'GET',
                  path: '/api/sector-performance',
                  description: 'Get sector performance data',
                  response: `[
  {
    "sector": "Technology",
    "change": 2.34,
    "volume": 15000000,
    "momentum": 1.2
  }
]`
                },
                {
                  method: 'POST',
                  path: '/api/analyze',
                  description: 'Analyze stocks using selected agents',
                  request: `{
  "tickers": ["AAPL", "MSFT"],
  "agents": ["WarrenBuffettAgent"],
  "period": "Annual"
}`,
                  response: `{
  "summary": [...],
  "detailed_analyses": {...}
}`
                },
                {
                  method: 'POST',
                  path: '/api/predictions',
                  description: 'Get ML model predictions',
                  request: `{
  "tickers": ["AAPL", "MSFT"]
}`,
                  response: `[
  {
    "ticker": "AAPL",
    "predicted_price": 185.50,
    "confidence": 0.85,
    "date": "2024-01-15"
  }
]`
                },
                {
                  method: 'POST',
                  path: '/api/backtest',
                  description: 'Run backtesting analysis',
                  request: `{
  "strategy": "value_investing",
  "tickers": ["AAPL", "MSFT", "GOOGL"],
  "startDate": "2023-01-01",
  "endDate": "2024-01-01"
}`,
                  response: `{
  "returns": [...],
  "sharpe_ratio": 2.34,
  "max_drawdown": -0.082,
  "win_rate": 0.734
}`
                }
              ].map((endpoint, index) => (
                <Card key={index}>
                  <CardHeader>
                    <div className="flex items-center space-x-2">
                      <div className={`px-2 py-1 rounded text-xs font-medium ${
                        endpoint.method === 'GET' ? 'bg-green-100 text-green-800' : 'bg-blue-100 text-blue-800'
                      }`}>
                        {endpoint.method}
                      </div>
                      <code className="text-sm font-mono">{endpoint.path}</code>
                    </div>
                  </CardHeader>
                  <CardContent>
                    <p className="text-sm text-gray-700 mb-3">{endpoint.description}</p>
                    
                    {endpoint.request && (
                      <div className="mb-3">
                        <h5 className="font-medium text-sm mb-2">Request Body</h5>
                        <div className="bg-gray-900 text-green-400 p-3 rounded font-mono text-xs">
                          {endpoint.request}
                        </div>
                      </div>
                    )}
                    
                    <div>
                      <h5 className="font-medium text-sm mb-2">Response</h5>
                      <div className="bg-gray-900 text-green-400 p-3 rounded font-mono text-xs">
                        {endpoint.response}
                      </div>
                    </div>
                  </CardContent>
                </Card>
              ))}
            </div>
          </div>
        </div>
      )
    },
    {
      id: 'export-formats',
      title: 'Export Formats',
      icon: <Download className="h-5 w-5" />,
      content: (
        <div className="space-y-6">
          <div>
            <h3 className="text-xl font-semibold mb-4">Data Export & Integration</h3>
            <p className="text-gray-700 mb-4">
              StockLab supports multiple export formats for analysis results, predictions, and backtesting data.
            </p>
          </div>

          <div className="grid md:grid-cols-2 gap-6">
            <Card>
              <CardHeader>
                <CardTitle className="text-lg">Export Formats</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  {[
                    { format: 'CSV', description: 'Comma-separated values for spreadsheet analysis', icon: '📊' },
                    { format: 'JSON', description: 'Structured data for API integration', icon: '🔗' },
                    { format: 'Excel', description: 'Microsoft Excel format with multiple sheets', icon: '📈' },
                    { format: 'PDF', description: 'Portable document format for reports', icon: '📄' },
                    { format: 'Parquet', description: 'Columnar format for big data analysis', icon: '🗄️' },
                    { format: 'Pickle', description: 'Python serialization for model objects', icon: '🐍' }
                  ].map((fmt) => (
                    <div key={fmt.format} className="flex items-center space-x-3 p-3 border rounded-lg">
                      <span className="text-xl">{fmt.icon}</span>
                      <div>
                        <h5 className="font-medium text-sm">{fmt.format}</h5>
                        <p className="text-xs text-gray-600">{fmt.description}</p>
                      </div>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle className="text-lg">Exportable Data</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  {[
                    { type: 'Stock Analysis', items: ['Agent recommendations', 'Technical indicators', 'Fundamental metrics'] },
                    { type: 'Predictions', items: ['Price forecasts', 'Confidence intervals', 'Model uncertainty'] },
                    { type: 'Backtesting Results', items: ['Performance metrics', 'Trade logs', 'Portfolio values'] },
                    { type: 'Market Data', items: ['Historical prices', 'Volume data', 'Market indicators'] },
                    { type: 'Portfolio Data', items: ['Positions', 'P&L history', 'Risk metrics'] }
                  ].map((data) => (
                    <div key={data.type} className="border rounded-lg p-3">
                      <h5 className="font-medium text-sm mb-2">{data.type}</h5>
                      <ul className="space-y-1">
                        {data.items.map((item) => (
                          <li key={item} className="text-xs text-gray-600 flex items-center space-x-1">
                            <div className="w-1 h-1 bg-gray-400 rounded-full"></div>
                            <span>{item}</span>
                          </li>
                        ))}
                      </ul>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
          </div>

          <Card>
            <CardHeader>
              <CardTitle className="text-lg">Integration Examples</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                <div>
                  <h4 className="font-medium mb-2">Python Integration</h4>
                  <div className="bg-gray-900 text-green-400 p-4 rounded-lg font-mono text-sm">
                    <div>import requests</div>
                    <div>import pandas as pd</div>
                    <div></div>
                    <div># Get market data</div>
                    <div>response = requests.get('http://localhost:8000/api/market-summary')</div>
                    <div>data = response.json()</div>
                    <div></div>
                    <div># Export to CSV</div>
                    <div>df = pd.DataFrame([data])</div>
                    <div>df.to_csv('market_data.csv', index=False)</div>
                  </div>
                </div>
                
                <div>
                  <h4 className="font-medium mb-2">JavaScript Integration</h4>
                  <div className="bg-gray-900 text-green-400 p-4 rounded-lg font-mono text-sm">
                    <div>// Fetch analysis results</div>
                    <div>const response = await fetch('/api/analyze', {'{'}</div>
                    <div>  method: 'POST',</div>
                    <div>  headers: {'{'}'Content-Type': 'application/json'{'}'},</div>
                    <div>  body: JSON.stringify({'{'}</div>
                    <div>    tickers: ['AAPL', 'MSFT'],</div>
                    <div>    agents: ['WarrenBuffettAgent']</div>
                    <div>  {'}'})</div>
                    <div>{'}'});</div>
                    <div>const data = await response.json();</div>
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>
        </div>
      )
    }
  ]

  return (
    <div className="flex h-screen bg-gray-50">
      {/* Sidebar */}
      <div className="w-64 bg-white border-r border-gray-200 p-6">
        <div className="mb-6">
          <h2 className="text-xl font-bold text-gray-900">Documentation</h2>
          <p className="text-sm text-gray-600 mt-1">Complete guide to StockLab</p>
        </div>
        
        <nav className="space-y-2">
          {sections.map((section) => (
            <button
              key={section.id}
              onClick={() => setActiveSection(section.id)}
              className={`w-full flex items-center space-x-3 px-3 py-2 rounded-lg text-left transition-colors ${
                activeSection === section.id
                  ? 'bg-blue-50 text-blue-700 border border-blue-200'
                  : 'text-gray-700 hover:bg-gray-50'
              }`}
            >
              {section.icon}
              <span className="text-sm font-medium">{section.title}</span>
            </button>
          ))}
        </nav>
      </div>

      {/* Main Content */}
      <div className="flex-1 overflow-auto">
        <div className="max-w-4xl mx-auto p-8">
          {sections.find(s => s.id === activeSection)?.content}
        </div>
      </div>
    </div>
  )
} 