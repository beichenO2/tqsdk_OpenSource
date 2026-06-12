import { useState } from 'react';
import { NavLink, Outlet } from 'react-router-dom';
import {
  LayoutDashboard, TrendingUp, Bot, FlaskConical,
  ShieldAlert, Settings, Trophy, Activity, Radio,
  Moon, Sun,
} from 'lucide-react';
import { cn } from '@/lib/cn';
import { useMarketStore, type Market } from '@/stores/marketStore';

const navItems = [
  { to: '/', icon: LayoutDashboard, label: '仪表盘' },
  { to: '/trading', icon: TrendingUp, label: '交易' },
  { to: '/strategies', icon: Bot, label: '策略' },
  { to: '/backtest', icon: FlaskConical, label: '回测' },
  { to: '/risk', icon: ShieldAlert, label: '风控' },
  { to: '/paper-trading', icon: Trophy, label: '模拟实盘' },
  { to: '/live-trading', icon: Radio, label: '实盘交易' },
];

const marketOptions: { value: Market; label: string }[] = [
  { value: 'all', label: '全部' },
  { value: 'futures', label: '期货' },
  { value: 'crypto', label: '加密' },
];

export default function Layout() {
  const [hovered, setHovered] = useState(false);
  const [dark, setDark] = useState(() => {
    if (typeof window !== 'undefined') {
      return localStorage.getItem('theme') === 'dark' ||
        (!localStorage.getItem('theme') && window.matchMedia('(prefers-color-scheme: dark)').matches);
    }
    return false;
  });
  const market = useMarketStore((s) => s.market);
  const setMarket = useMarketStore((s) => s.setMarket);

  const toggleDark = () => {
    const next = !dark;
    setDark(next);
    document.documentElement.classList.toggle('dark', next);
    localStorage.setItem('theme', next ? 'dark' : 'light');
  };

  useState(() => {
    document.documentElement.classList.toggle('dark', dark);
  });

  const expanded = hovered;

  return (
    <div className="flex h-screen">
      <aside
        onMouseEnter={() => setHovered(true)}
        onMouseLeave={() => setHovered(false)}
        className={cn(
          'fixed top-0 left-0 h-screen z-40 flex shrink-0 flex-col bg-surface-secondary border-r border-border overflow-hidden transition-all duration-300 ease-out',
          expanded ? 'w-[14rem]' : 'w-[4rem]',
        )}
      >
        {/* Logo */}
        <div className="flex items-center px-4 h-[4rem] border-b border-border shrink-0">
          <span className={cn(
            'font-bold text-brand whitespace-nowrap overflow-hidden transition-all duration-300',
            expanded ? 'text-lg' : 'text-xl w-8 text-center',
          )}>
            {expanded ? 'PolarTrade' : 'P'}
          </span>
        </div>

        {/* Market switcher */}
        <div className="px-3 py-4 border-b border-border">
          {expanded ? (
            <div className="flex rounded-xl bg-surface-tertiary p-1 gap-0.5">
              {marketOptions.map((opt) => (
                <button
                  key={opt.value}
                  type="button"
                  onClick={() => setMarket(opt.value)}
                  className={cn(
                    'flex-1 rounded-lg px-3 py-2 text-[13px] font-medium transition-all',
                    market === opt.value
                      ? 'bg-white text-brand shadow-sm'
                      : 'text-text-muted hover:text-text-secondary',
                  )}
                >
                  {opt.label}
                </button>
              ))}
            </div>
          ) : (
            <div className="flex items-center justify-center py-1">
              <span className="text-[11px] font-semibold text-brand">
                {market === 'all' ? 'A' : market === 'futures' ? '期' : '币'}
              </span>
            </div>
          )}
        </div>

        {/* Navigation */}
        <nav className="flex flex-1 flex-col gap-1 px-3 py-4 overflow-y-auto">
          {navItems.map(({ to, icon: Icon, label }) => (
            <NavLink
              key={to}
              to={to}
              end={to === '/'}
              className={({ isActive }) =>
                cn(
                  'group flex items-center rounded-xl transition-all duration-200 whitespace-nowrap min-h-[44px]',
                  expanded ? 'gap-3 px-4 py-3' : 'justify-center px-0 py-3',
                  isActive
                    ? 'bg-brand/8 text-brand font-medium'
                    : 'text-text-secondary hover:bg-surface-tertiary hover:text-text-primary',
                )
              }
            >
              <Icon className="h-5 w-5 shrink-0" />
              {expanded && <span className="text-[15px]">{label}</span>}
            </NavLink>
          ))}
        </nav>

        {/* Status + Settings */}
        <div className="border-t border-border px-3 py-4 space-y-1">
          <div className={cn('flex items-center rounded-xl px-4 py-2', expanded ? 'gap-2.5' : 'justify-center')}>
            <Activity className="h-4 w-4 text-profit shrink-0" />
            {expanded && <span className="text-[13px] text-text-muted">在线</span>}
          </div>
          <button
            onClick={toggleDark}
            className={cn(
              'flex items-center rounded-xl transition-all duration-200 whitespace-nowrap min-h-[44px] w-full',
              expanded ? 'gap-3 px-4 py-3' : 'justify-center px-0 py-3',
              'text-text-secondary hover:bg-surface-tertiary hover:text-text-primary',
            )}
          >
            {dark ? <Sun className="h-5 w-5 shrink-0" /> : <Moon className="h-5 w-5 shrink-0" />}
            {expanded && <span className="text-[15px]">{dark ? '浅色' : '深色'}</span>}
          </button>
          <NavLink
            to="/settings"
            className={({ isActive }) =>
              cn(
                'flex items-center rounded-xl transition-all duration-200 whitespace-nowrap min-h-[44px]',
                expanded ? 'gap-3 px-4 py-3' : 'justify-center px-0 py-3',
                isActive
                  ? 'bg-brand/8 text-brand font-medium'
                  : 'text-text-secondary hover:bg-surface-tertiary hover:text-text-primary',
              )
            }
          >
            <Settings className="h-5 w-5 shrink-0" />
            {expanded && <span className="text-[15px]">设置</span>}
          </NavLink>
        </div>
      </aside>

      <main className={cn(
        'flex-1 overflow-y-auto bg-surface-950 transition-[margin-left] duration-300 ease-out min-h-screen',
        'ml-[4rem]',
      )}>
        <Outlet />
      </main>
    </div>
  );
}
