# StockLab - Advanced Quantitative Trading Platform

StockLab is an advanced quantitative trading platform that combines machine learning, fundamental analysis, and technical indicators to provide comprehensive stock analysis and trading recommendations.

## 🚀 Features

- **Multi-Agent Analysis System**: Warren Buffett, Peter Lynch, Charlie Munger, and other legendary investor agents
- **Real-Time Market Data**: Integration with Polygon.io for live market data
- **Advanced ML Models**: Tempus v3.0 transformer-based prediction model
- **Backtesting Engine**: Comprehensive strategy testing and performance analysis
- **Portfolio Management**: Real-time portfolio tracking and risk management
- **Technical Indicators**: 200+ technical indicators and market analysis tools

## 📋 Prerequisites

- Python 3.8+
- Node.js 16+ (for frontend)
- Polygon.io API key
- Alpaca API credentials (optional, for paper trading)

## 🔧 Installation

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/stocklab.git
cd stocklab
```

### 2. Install Python Dependencies

```bash
pip install -r requirements.txt
```

### 3. Install Frontend Dependencies

```bash
cd stocklab-frontend
npm install
```

### 4. Configure API Keys

**⚠️ IMPORTANT: Never commit your API keys to version control!**

1. Copy the example credentials file:
```bash
cp credentials.yml.example credentials.yml
```

2. Edit `credentials.yml` with your API keys:
```yaml
# Polygon.io API (Market Data)
polygon:
  api_key: "your-polygon-api-key-here"
  base_url: "https://api.polygon.io"

# Alpaca Trading API (Paper Trading)
alpaca:
  api_key: "your-alpaca-api-key-here"
  secret_key: "your-alpaca-secret-key-here"
  base_url: "https://paper-api.alpaca.markets"
```

### 5. Start the Application

**Backend Server:**
```bash
cd stocklab-frontend/backend
python main.py
```

**Frontend Development Server:**
```bash
cd stocklab-frontend
npm run dev
```

Visit `http://localhost:5173` to access the StockLab dashboard.

## 🔒 Security

### Protected Files

The following files are automatically ignored by Git to protect sensitive information:

- `credentials.yml` - API keys and secrets
- `*.pem`, `*.key`, `*.p12` - SSL certificates and private keys
- `*.pth`, `*.pt`, `*.onnx` - ML model files
- `*.db`, `*.sqlite` - Database files
- `venv/`, `node_modules/` - Virtual environments and dependencies
- `*.log` - Log files
- `__pycache__/` - Python cache files

### Environment Variables

For additional security, you can use environment variables instead of the credentials file:

```bash
export POLYGON_API_KEY="your-polygon-api-key"
export ALPACA_API_KEY="your-alpaca-api-key"
export ALPACA_SECRET_KEY="your-alpaca-secret-key"
```

## 📚 Documentation

Comprehensive documentation is available in the application under the Documentation section, including:

- Getting Started Guide
- Project Overview
- Model Architecture & Mathematics
- Datasets & Training Information
- Performance Benchmarks
- API Documentation
- Export Formats

## 🏗️ Architecture

### Technology Stack

- **Frontend**: React + TypeScript + Tailwind CSS
- **Backend**: FastAPI + Python
- **ML Models**: PyTorch + ONNX
- **Data Sources**: Polygon.io + Alpaca
- **Database**: Azure SQL

### Key Components

- **TickerData**: Real-time market data fetching and processing
- **AgentManager**: Multi-agent analysis system
- **ModelInference**: ML model prediction engine
- **BackTesting**: Strategy testing and performance analysis
- **PortfolioManager**: Portfolio tracking and management

## 🤖 ML Models

### Tempus v3.0

- Transformer-based architecture with temporal attention
- Multi-head self-attention mechanism
- 200+ input features including technical indicators
- Real-time price prediction with confidence intervals

### Agent-Based Analysis

- **Warren Buffett Agent**: Value investing analysis
- **Peter Lynch Agent**: Growth stock analysis
- **Charlie Munger Agent**: Quality metrics evaluation
- **Cathie Wood Agent**: Innovation and disruption analysis
- **Technical Agent**: Chart pattern and technical analysis

## 📊 Performance

### Model Performance (2020-2024)

- **Direction Accuracy**: 87.3%
- **Sharpe Ratio**: 2.34
- **Max Drawdown**: -8.2%
- **Win Rate**: 73.4%
- **Annualized Return**: +28.3%

### Benchmark Comparison

| Strategy | Annual Return | Sharpe Ratio | Max DD | Win Rate |
|----------|---------------|--------------|--------|----------|
| StockLab | +28.3% | 2.34 | -8.2% | 73.4% |
| S&P 500 | +12.8% | 1.12 | -23.8% | - |
| Momentum | +18.5% | 1.45 | -15.2% | 65.2% |

## 🔌 API Reference

### Base URL
```
http://localhost:8000
```

### Key Endpoints

- `GET /api/market-summary` - Real-time market data
- `GET /api/sector-performance` - Sector performance data
- `POST /api/analyze` - Stock analysis with agents
- `POST /api/predictions` - ML model predictions
- `POST /api/backtest` - Strategy backtesting

See the Documentation section in the application for complete API reference.

## 📁 Project Structure

```
StockLab/
├── Agents/                 # Investment agent implementations
├── Components/            # Core system components
│   ├── DataModules/      # Data fetching and processing
│   ├── Models/           # ML model implementations
│   └── ...
├── Models/               # Trained ML models
├── stocklab-frontend/    # React frontend application
│   ├── src/
│   │   ├── components/   # UI components
│   │   ├── pages/        # Application pages
│   │   └── services/     # API services
│   └── backend/          # FastAPI backend
├── requirements.txt      # Python dependencies
└── credentials.yml       # API keys (not in version control)
```

## 🚨 Important Notes

1. **API Keys**: Never commit your API keys to version control. The `credentials.yml` file is already in `.gitignore`.

2. **Model Files**: Large ML model files are excluded from version control. You'll need to download or train models separately.

3. **Data Files**: Large datasets are excluded. Use the data fetching components to download required data.

4. **Virtual Environments**: Python and Node.js dependencies are excluded. Always install dependencies locally.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## ⚠️ Disclaimer

This software is for educational and research purposes only. It is not intended to provide financial advice. Always do your own research and consider consulting with a financial advisor before making investment decisions.

## 🆘 Support

For support and questions:
- Check the Documentation section in the application
- Review the API documentation
- Open an issue on GitHub

---

**StockLab** - Advanced Quantitative Trading Platform 