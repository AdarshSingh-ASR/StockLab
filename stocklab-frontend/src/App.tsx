import React from 'react'
import { Layout } from './components/Layout'
import { Dashboard } from './pages/Dashboard'
import { StockAnalysis } from './pages/StockAnalysis'
import { PortfolioPage } from './pages/Portfolio'
import { PredictionsPage } from './pages/Predictions'
import { AgentsPage } from './pages/Agents'
import { BacktestingPage } from './pages/Backtesting'
import { MarketData } from './pages/MarketData'
import { Documentation } from './pages/Documentation'

// Simple routing based on pathname
const getCurrentPage = () => {
  const pathname = window.location.pathname
  switch (pathname) {
    case '/':
      return <Dashboard />
    case '/analysis':
      return <StockAnalysis />
    case '/portfolio':
      return <PortfolioPage />
    case '/predictions':
      return <PredictionsPage />
    case '/agents':
      return <AgentsPage />
    case '/backtesting':
      return <BacktestingPage />
    case '/market':
      return <MarketData />
    case '/settings':
      return <Documentation />
    default:
      return <Dashboard />
  }
}

function App() {
  const [currentPage, setCurrentPage] = React.useState(getCurrentPage)

  React.useEffect(() => {
    const handlePopState = () => {
      setCurrentPage(getCurrentPage())
    }

    window.addEventListener('popstate', handlePopState)
    return () => window.removeEventListener('popstate', handlePopState)
  }, [])

  return (
    <Layout>
      {currentPage}
    </Layout>
  )
}

export default App
