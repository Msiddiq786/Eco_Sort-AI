import { useEffect, useState, type ReactNode } from 'react'
import { Link } from 'react-router-dom'
import { Menu } from 'lucide-react'
import Sidebar from './Sidebar'

interface LayoutProps {
  children: ReactNode
}

export default function Layout({ children }: LayoutProps) {
  const [mobileOpen, setMobileOpen] = useState(false)

  useEffect(() => {
    const onKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape') setMobileOpen(false)
    }
    window.addEventListener('keydown', onKeyDown)
    return () => window.removeEventListener('keydown', onKeyDown)
  }, [])

  return (
    <div className="min-h-screen bg-ecosort-bg flex flex-col lg:flex-row">
      {/* Mobile Top Bar */}
      <header className="lg:hidden sticky top-0 z-30 flex items-center justify-between px-4 py-3 bg-white/95 backdrop-blur border-b border-slate-200/80 shadow-eco-sm">
        <Link to="/dashboard" className="flex items-center gap-2.5">
          <img
            src="/logo-icon.svg"
            alt="EcoSort AI"
            className="h-8 w-8 object-contain"
          />
          <div>
            <span className="font-heading font-extrabold text-sm tracking-tight text-ecosort-forest">
              EcoSort
            </span>{' '}
            <span className="badge-lime text-[10px] py-0 px-1.5">AI</span>
          </div>
        </Link>

        <button
          type="button"
          onClick={() => setMobileOpen(true)}
          aria-label="Open navigation menu"
          className="p-2 rounded-xl border border-slate-200 bg-white text-slate-700 hover:bg-slate-50"
        >
          <Menu className="h-5 w-5" />
        </button>
      </header>

      <Sidebar
        mobileOpen={mobileOpen}
        onCloseMobile={() => setMobileOpen(false)}
      />

      <main className="flex-1 min-w-0">
        <div className="px-4 sm:px-6 lg:px-8 py-6 max-w-7xl mx-auto animate-fade-in">
          {children}
        </div>
      </main>
    </div>
  )
}
