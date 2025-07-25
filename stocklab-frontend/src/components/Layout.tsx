import React from 'react'
import { 
  BarChart3, 
  TrendingUp, 
  Users, 
  Brain, 
  BookOpen, 
  Home,
  Activity,
  PieChart
} from 'lucide-react'
import { cn } from '../lib/utils'

interface LayoutProps {
  children: React.ReactNode
}

const navigation = [
  { name: 'Dashboard', href: '/', icon: Home },
  { name: 'Stock Analysis', href: '/analysis', icon: BarChart3 },
  { name: 'Portfolio', href: '/portfolio', icon: PieChart },
  { name: 'Predictions', href: '/predictions', icon: TrendingUp },
  { name: 'Agents', href: '/agents', icon: Brain },
  { name: 'Backtesting', href: '/backtesting', icon: Activity },
  { name: 'Market Data', href: '/market', icon: Users },
  { name: 'Documentation', href: '/settings', icon: BookOpen },
]

export const Layout: React.FC<LayoutProps> = ({ children }) => {
  const [currentPath, setCurrentPath] = React.useState('/')

  React.useEffect(() => {
    setCurrentPath(window.location.pathname)
  }, [])

  return (
    <div className="min-h-screen bg-background">
      {/* Sidebar */}
      <div className="fixed inset-y-0 left-0 z-50 w-64 bg-card border-r">
        <div className="flex h-full flex-col">
          {/* Logo */}
          <div className="flex h-16 items-center px-6 border-b">
            <div className="flex items-center space-x-2">
              <div className="w-8 h-8 bg-primary rounded-lg flex items-center justify-center">
                <TrendingUp className="w-5 h-5 text-primary-foreground" />
              </div>
              <span className="text-lg font-semibold">StockLab</span>
            </div>
          </div>

          {/* Navigation */}
          <nav className="flex-1 space-y-1 px-3 py-4">
            {navigation.map((item) => {
              const isActive = currentPath === item.href
              return (
                <a
                  key={item.name}
                  href={item.href}
                  className={cn(
                    'group flex items-center px-3 py-2 text-sm font-medium rounded-md transition-colors',
                    isActive
                      ? 'bg-primary text-primary-foreground'
                      : 'text-muted-foreground hover:text-foreground hover:bg-accent'
                  )}
                  onClick={() => setCurrentPath(item.href)}
                >
                  <item.icon
                    className={cn(
                      'mr-3 h-4 w-4 flex-shrink-0',
                      isActive ? 'text-primary-foreground' : 'text-muted-foreground group-hover:text-foreground'
                    )}
                  />
                  {item.name}
                </a>
              )
            })}
          </nav>

          {/* User section */}
          <div className="border-t p-4">
            <div className="flex items-center space-x-3">
              <div className="w-8 h-8 bg-muted rounded-full flex items-center justify-center">
                <span className="text-sm font-medium">U</span>
              </div>
              <div className="flex-1 min-w-0">
                <p className="text-sm font-medium text-foreground truncate">
                  User
                </p>
                <p className="text-xs text-muted-foreground truncate">
                  user@example.com
                </p>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Main content */}
      <div className="pl-64">
        <main className="min-h-screen">
          {children}
        </main>
      </div>
    </div>
  )
} 