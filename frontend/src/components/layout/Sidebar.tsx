import { useEffect, useState } from 'react'
import { NavLink, Link } from 'react-router-dom'
import {
  LayoutDashboard,
  ScanLine,
  Crosshair,
  History,
  BarChart3,
  BookOpen,
  Award,
  Compass,
  Leaf,
  Sliders,
  Sparkles,
  X,
} from 'lucide-react'
import { cn } from '@/lib/utils'
import { getHealth, getEnsembleConfig } from '@/services/api'

interface NavGroup {
  title: string
  items: {
    to: string
    label: string
    icon: typeof LayoutDashboard
    badge?: string
  }[]
}

const NAV_GROUPS: NavGroup[] = [
  {
    title: 'Core AI Pipeline',
    items: [
      { to: '/dashboard', label: 'Dashboard', icon: LayoutDashboard },
      { to: '/classify', label: 'Classify Waste', icon: ScanLine },
      { to: '/detect', label: 'Detect Objects', icon: Crosshair, badge: 'V3' },
      { to: '/history', label: 'History & Audit', icon: History },
    ],
  },
  {
    title: 'Research & Ensemble',
    items: [
      { to: '/performance', label: 'Model Metrics', icon: BarChart3 },
      { to: '/ensemble', label: 'Ensemble Analysis', icon: Sliders },
    ],
  },
  {
    title: 'Sustainability Hub',
    items: [
      { to: '/guide', label: 'Recycling Guide', icon: Compass },
      { to: '/learning', label: 'Learning Center', icon: BookOpen },
      { to: '/impact', label: 'Impact Tracker', icon: Leaf },
      { to: '/rewards', label: 'Eco Rewards', icon: Award },
    ],
  },
]

interface SidebarProps {
  mobileOpen?: boolean
  onCloseMobile?: () => void
}

export default function Sidebar({ mobileOpen = false, onCloseMobile }: SidebarProps) {
  const [modelsOnline, setModelsOnline] = useState<boolean | null>(null)
  const [fusionSplit, setFusionSplit] = useState<{ vit: number; clip: number }>({
    vit: 70,
    clip: 30,
  })

  useEffect(() => {
    let active = true
    Promise.allSettled([getHealth(), getEnsembleConfig()]).then(([healthRes, cfgRes]) => {
      if (!active) return
      if (healthRes.status === 'fulfilled') {
        setModelsOnline(healthRes.value.status === 'ok' || healthRes.value.models_ready)
      } else {
        setModelsOnline(false)
      }
      if (cfgRes.status === 'fulfilled') {
        setFusionSplit({
          vit: Math.round(cfgRes.value.vit_weight * 100),
          clip: Math.round(cfgRes.value.clip_weight * 100),
        })
      }
    })
    return () => {
      active = false
    }
  }, [])

  const navContent = (
    <div className="flex flex-col h-full bg-white border-r border-slate-200/80 shadow-eco-sm">
      {/* Brand Header */}
      <div className="flex items-center justify-between px-5 py-4 border-b border-slate-100">
        <Link
          to="/dashboard"
          onClick={onCloseMobile}
          className="flex items-center group focus:outline-none"
          aria-label="EcoSort AI Dashboard"
        >
          <img
            src="/logo.svg"
            alt="EcoSort AI — Smart Waste Segregation"
            className="h-10 w-auto object-contain transition-transform duration-150 group-hover:scale-[1.01]"
          />
        </Link>
        {onCloseMobile && (
          <button
            type="button"
            onClick={onCloseMobile}
            aria-label="Close navigation menu"
            className="lg:hidden p-1.5 rounded-lg text-slate-400 hover:bg-slate-100 hover:text-slate-700"
          >
            <X className="h-5 w-5" />
          </button>
        )}
      </div>

      {/* Grouped Navigation */}
      <nav aria-label="Primary Navigation" className="flex-1 px-3 py-4 space-y-5 overflow-y-auto scrollbar-thin">
        {NAV_GROUPS.map((group) => (
          <div key={group.title} className="space-y-1">
            <p className="px-3 text-[10px] font-bold uppercase tracking-wider text-slate-400">
              {group.title}
            </p>
            <div className="space-y-0.5">
              {group.items.map(({ to, label, icon: Icon, badge }) => (
                <NavLink
                  key={to}
                  to={to}
                  onClick={onCloseMobile}
                  className={({ isActive }) =>
                    cn(
                      'flex items-center justify-between px-3 py-2 rounded-xl text-sm font-medium transition-all duration-150',
                      isActive
                        ? 'bg-ecosort-emerald text-white shadow-eco-sm font-semibold'
                        : 'text-slate-600 hover:bg-ecosort-mint/80 hover:text-ecosort-forest'
                    )
                  }
                >
                  {({ isActive }) => (
                    <>
                      <div className="flex items-center gap-2.5 min-w-0">
                        <Icon className="h-4 w-4 shrink-0" />
                        <span className="truncate">{label}</span>
                      </div>
                      {badge && (
                        <span
                          className={cn(
                            'text-[10px] font-extrabold px-1.5 py-0.5 rounded-full uppercase tracking-wider',
                            isActive
                              ? 'bg-white/20 text-white'
                              : 'bg-ecosort-lime text-ecosort-forest'
                          )}
                        >
                          {badge}
                        </span>
                      )}
                    </>
                  )}
                </NavLink>
              ))}
            </div>
          </div>
        ))}
      </nav>

      {/* Pipeline Status Card */}
      <div className="p-3.5 mx-3 mb-3 rounded-2xl bg-ecosort-sage border border-emerald-200/70">
        <div className="flex items-center justify-between gap-2 mb-1">
          <div className="flex items-center gap-1.5">
            <Sparkles className="h-3.5 w-3.5 text-ecosort-emerald" />
            <p className="text-xs font-heading font-bold text-ecosort-forest">
              Dual-Model Ensemble
            </p>
          </div>
          <span
            className={cn(
              'h-2 w-2 rounded-full shrink-0',
              modelsOnline === null
                ? 'bg-slate-300 animate-pulse'
                : modelsOnline
                  ? 'bg-emerald-500'
                  : 'bg-amber-500'
            )}
            title={
              modelsOnline === null
                ? 'Checking backend status…'
                : modelsOnline
                  ? 'API & Models Online'
                  : 'API Offline or Starting'
            }
          />
        </div>
        <p className="text-[11px] text-slate-600 leading-snug">
          {fusionSplit.vit}% ViT-Base + {fusionSplit.clip}% CLIP V2
        </p>
      </div>

      {/* Footer */}
      <div className="px-5 py-3 border-t border-slate-100 flex items-center justify-between text-[11px] text-slate-400">
        <span className="font-medium">EcoSort AI · V3</span>
        <span className={cn('font-medium', modelsOnline ? 'text-emerald-700' : 'text-slate-400')}>
          {modelsOnline === null ? 'Connecting…' : modelsOnline ? 'API Ready' : 'Offline Mode'}
        </span>
      </div>
    </div>
  )

  return (
    <>
      {/* Desktop sticky sidebar */}
      <aside className="hidden lg:block w-64 shrink-0 sticky top-0 h-screen">
        {navContent}
      </aside>

      {/* Mobile slide-over drawer */}
      {mobileOpen && (
        <div className="fixed inset-0 z-50 lg:hidden flex">
          <div
            className="fixed inset-0 bg-slate-900/50 backdrop-blur-sm"
            onClick={onCloseMobile}
            aria-hidden="true"
          />
          <aside className="relative w-72 max-w-[85vw] h-full z-10 animate-fade-in">
            {navContent}
          </aside>
        </div>
      )}
    </>
  )
}
