# StockLab Frontend

A modern React + Vite frontend for the StockLab investment analysis platform. This frontend provides a sleek, functional interface that integrates with the existing StockLab backend components including AI agents, ML models, and data analysis tools.

## Features

- **Dashboard**: Overview of portfolio performance, market data, and recent activity
- **Stock Analysis**: Multi-agent analysis using different investment philosophies (Warren Buffett, Peter Lynch, etc.)
- **Portfolio Management**: Track positions, P&L, and portfolio performance
- **Predictions**: ML model price predictions and forecasts
- **Backtesting**: Strategy backtesting and performance analysis
- **Market Data**: Real-time market data and technical indicators
- **Agent Management**: Configure and manage AI analysis agents

## Tech Stack

- **Frontend**: React 18 + TypeScript + Vite
- **Styling**: Tailwind CSS with custom design system
- **Icons**: Lucide React
- **Charts**: Recharts (for future implementation)
- **Backend**: FastAPI (Python)
- **State Management**: React hooks and context

## Design Philosophy

The UI is inspired by Linear and Notion with:
- Minimal, spacious design
- Small font sizes and clean typography
- Subtle shadows and borders
- Consistent color scheme
- Responsive layout

## Quick Start

### Prerequisites

- Node.js 18+ 
- Python 3.8+
- Access to StockLab backend components

### Frontend Setup

1. Install dependencies:
```bash
npm install
```

2. Start the development server:
```bash
npm run dev
```

The frontend will be available at `http://localhost:5173`

### Backend Setup

1. Navigate to the backend directory:
```bash
cd backend
```

2. Install Python dependencies:
```bash
pip install -r requirements.txt
```

3. Start the FastAPI server:
```bash
python main.py
```

The API will be available at `http://localhost:8000`

### Production Build

1. Build the frontend:
```bash
npm run build
```

2. Preview the production build:
```bash
npm run preview
```

## Project Structure

```
stocklab-frontend/
├── src/
│   ├── components/
│   │   ├── ui/           # Reusable UI components
│   │   └── Layout.tsx    # Main layout component
│   ├── pages/            # Page components
│   ├── services/         # API services
│   ├── types/            # TypeScript type definitions
│   ├── lib/              # Utility functions
│   └── App.tsx           # Main app component
├── backend/
│   ├── main.py           # FastAPI server
│   └── requirements.txt  # Python dependencies
└── public/               # Static assets
```

## API Integration

The frontend connects to the StockLab backend through a FastAPI server that provides:

- **Stock Analysis**: Multi-agent analysis using existing StockLab agents
- **Data Fetching**: Historical stock data and technical indicators
- **ML Predictions**: Integration with Tempus models
- **Portfolio Management**: Position tracking and P&L calculation
- **Backtesting**: Strategy testing and performance analysis

## Key Components

### Agent System
The frontend integrates with all StockLab agents:
- Warren Buffett (Value investing)
- Peter Lynch (Growth at reasonable price)
- Charlie Munger (Mental models)
- Cathie Wood (Innovation and disruption)
- Bill Ackman (Concentrated positions)
- Stanley Druckenmiller (Macro trends)
- Ben Graham (Security analysis)
- Phil Fisher (Qualitative analysis)
- Aswath Damodaran (Valuation)
- Valuation Agent (Multi-factor)
- Fundamentals Agent (Financial metrics)

### ML Models
Integration with StockLab's neural network models:
- Tempus v2/v3 for price prediction
- HMM for market regime detection
- Technical indicators and feature engineering

### Data Pipeline
Real-time data from:
- Polygon.io for market data
- Financial statements and ratios
- Technical indicators
- News and sentiment analysis

## Development

### Adding New Pages

1. Create a new component in `src/pages/`
2. Add the route to `src/App.tsx`
3. Update the navigation in `src/components/Layout.tsx`

### Styling

The project uses Tailwind CSS with a custom design system:
- CSS variables for theming
- Consistent spacing and typography
- Responsive design patterns
- Dark mode support (future)

### API Development

The backend uses FastAPI with:
- Automatic API documentation at `/docs`
- CORS support for frontend integration
- Pydantic models for data validation
- Mock data for development

## Contributing

1. Follow the existing code style and patterns
2. Use TypeScript for type safety
3. Add proper error handling
4. Test API integration
5. Update documentation

## License

This project is part of the StockLab investment analysis platform.

## Support

For issues and questions, please refer to the main StockLab repository documentation.
