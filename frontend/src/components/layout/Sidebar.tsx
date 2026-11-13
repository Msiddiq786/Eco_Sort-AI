import { NavLink } from 'react-router-dom'
import {
  LayoutDashboard,
  ScanLine,
  Crosshair,
  History,
  BarChart3,
  Recycle,
  Sparkles,
  BookOpen,
  Award,
  Compass,
  Leaf,
} from 'lucide-react'
import { cn } from '@/lib/utils'

const navItems = [
  { to: '/dashboard',   label: 'Dashboard / Analytics', icon: LayoutDashboard },
  { to: '/classify',    label: 'Classify Waste',        icon: ScanLine },
  { to: '/detect',      label: 'Detect Objects',        icon: Crosshair, badge: 'Beta' },
  { to: '/learning',    label: 'Learning Center',       icon: BookOpen },
  { to: '/rewards',     label: 'Eco Rewards',           icon: Award },
  { to: '/guide',       label: 'Smart Recycling Guide', icon: Compass },
  { to: '/impact',      label: 'Impact Tracker',        icon: Leaf },
  { to: '/history',     label: 'History & Audit',       icon: History },
  { to: '/performance', label: 'Model Metrics',         icon: BarChart3 },
]

export default function Sidebar() {
  return (
    <aside className="flex flex-col w-64 min-h-screen bg-white border-r border-slate-200/80 shadow-sm">
      {/* Brand Header */}
      <div className="flex items-center gap-3 px-6 py-5 border-b border-slate-100">
        <div className="p-2.5 rounded-xl bg-[#14532D] text-[#D9F99D] shadow-sm">
          <Recycle className="h-5 w-5 text-[#D9F99D]" />
        </div>
        <div>
          <div className="flex items-center gap-1.5">
            <span className="font-heading font-extrabold text-base tracking-tight text-[#14532D]">
              EcoSort
            </span>
            <span className="inline-flex items-center text-[10px] font-bold px-1.5 py-0.5 rounded-full bg-[#D9F99D] text-[#14532D]">
              AI
            </span>
          </div>
          <p className="text-[11px] text-slate-500 font-medium">Vision Transformer Ensemble</p>
        </div>
      </div>

      {/* Navigation */}
      <nav className="flex-1 px-3 py-5 space-y-1">
        {navItems.map(({ to, label, icon: Icon, badge }) => (
          <NavLink
            key={to}
            to={to}
            className={({ isActive }) =>
              cn(
                'flex items-center justify-between px-3.5 py-2.5 rounded-lg text-sm font-medium transition-all duration-150',
                isActive
                  ? 'bg-[#16A34A] text-white shadow-sm'
                  : 'text-slate-600 hover:bg-[#F1F5F0] hover:text-[#14532D]'
              )
            }
          >
            <div className="flex items-center gap-3">
              <Icon className="h-4 w-4 shrink-0" />
              <span>{label}</span>
            </div>
            {badge && (
              <span className="text-[10px] font-bold px-1.5 py-0.5 rounded-full bg-[#D9F99D] text-[#14532D] uppercase tracking-wider">
                {badge}
              </span>
            )}
          </NavLink>
        ))}
      </nav>

      {/* Pipeline Status Pill */}
      <div className="p-4 mx-3 mb-3 rounded-xl bg-[#F7F9F5] border border-slate-200/60">
        <div className="flex items-center gap-2 mb-1">
          <Sparkles className="h-3.5 w-3.5 text-[#16A34A]" />
          <p className="text-xs font-heading font-semibold text-[#14532D]">Dual-Model Pipeline</p>
        </div>
        <p className="text-[11px] text-slate-500 leading-snug">
          ViT (70%) + CLIP Foundation Model (30%)
        </p>
      </div>

      {/* Footer */}
      <div className="px-6 py-3.5 border-t border-slate-100 flex items-center justify-between text-[11px] text-slate-400">
        <span>Production v1.0.0</span>
        <span className="inline-block h-2 w-2 rounded-full bg-emerald-500" title="System Online" />
      </div>
    </aside>
  )
}
