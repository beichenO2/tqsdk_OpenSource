import { useEffect, useState } from 'react';
import { Link, NavLink, Outlet, useLocation } from 'react-router-dom';
import {
  FlaskConical, TrendingUp, ShieldAlert, Settings2,
  Moon, Sun, Search, Gauge, ChevronRight, Zap,
} from 'lucide-react';
import { useQuery } from '@tanstack/react-query';
import {
  Sidebar,
  SidebarContent,
  SidebarFooter,
  SidebarGroup,
  SidebarGroupContent,
  SidebarGroupLabel,
  SidebarHeader,
  SidebarInset,
  SidebarMenu,
  SidebarMenuButton,
  SidebarMenuItem,
  SidebarMenuSub,
  SidebarMenuSubButton,
  SidebarMenuSubItem,
  SidebarProvider,
  SidebarRail,
  SidebarTrigger,
} from '@/components/shadcn/sidebar';
import {
  Breadcrumb,
  BreadcrumbItem,
  BreadcrumbList,
  BreadcrumbPage,
  BreadcrumbSeparator,
} from '@/components/shadcn/breadcrumb';
import { Separator } from '@/components/shadcn/separator';
import { Badge } from '@/components/shadcn/badge';
import { Button } from '@/components/shadcn/button';
import { cn } from '@/lib/utils';
import { useMarketStore, type Market } from '@/stores/marketStore';
import { WORKSPACES, workspaceForPath, type WorkspaceId } from '@/nav/workspaces';
import { CommandPalette } from '@/components/CommandPalette';
import { api } from '@/services/api';

const WS_ICONS: Record<WorkspaceId, typeof FlaskConical> = {
  research: FlaskConical,
  trading: TrendingUp,
  monitor: ShieldAlert,
  platform: Settings2,
};

const marketOptions: { value: Market; label: string }[] = [
  { value: 'all', label: '全部' },
  { value: 'futures', label: '期货' },
  { value: 'crypto', label: '加密' },
];

function useSystemHealth() {
  return useQuery({
    queryKey: ['system-health-shell'],
    queryFn: () => api.getSystemHealth(),
    refetchInterval: 15_000,
    retry: 1,
  });
}

function AppSidebar() {
  const location = useLocation();
  const { data: health } = useSystemHealth();
  const components = (health?.components || {}) as Record<string, { ok?: boolean }>;
  const gwOk = !!components.tqsdk_gateway?.ok;
  const liveEnabled = !!(health as { live_enabled?: boolean } | undefined)?.live_enabled;

  return (
    <Sidebar collapsible="icon" variant="inset">
      <SidebarHeader>
        <SidebarMenu>
          <SidebarMenuItem>
            <SidebarMenuButton size="lg" asChild>
              <Link to="/">
                <div className="bg-primary text-primary-foreground flex aspect-square size-8 items-center justify-center rounded-lg">
                  <Zap className="size-4" />
                </div>
                <div className="grid flex-1 text-left leading-tight">
                  <span className="truncate font-semibold">PolarTrade</span>
                  <span className="truncate text-xs text-muted-foreground">
                    tqsdk quant desk
                  </span>
                </div>
              </Link>
            </SidebarMenuButton>
          </SidebarMenuItem>
        </SidebarMenu>
      </SidebarHeader>

      <SidebarContent>
        <SidebarGroup>
          <SidebarGroupLabel>总览</SidebarGroupLabel>
          <SidebarGroupContent>
            <SidebarMenu>
              <SidebarMenuItem>
                <SidebarMenuButton
                  asChild
                  isActive={location.pathname === '/'}
                  tooltip="仪表盘"
                >
                  <NavLink to="/">
                    <Gauge />
                    <span>仪表盘</span>
                  </NavLink>
                </SidebarMenuButton>
              </SidebarMenuItem>
            </SidebarMenu>
          </SidebarGroupContent>
        </SidebarGroup>

        <SidebarGroup>
          <SidebarGroupLabel>工作区</SidebarGroupLabel>
          <SidebarGroupContent>
            <SidebarMenu>
              {WORKSPACES.map((ws) => {
                const Icon = WS_ICONS[ws.id];
                const active = workspaceForPath(location.pathname) === ws.id
                  && location.pathname !== '/';
                return (
                  <SidebarMenuItem key={ws.id}>
                    <SidebarMenuButton asChild isActive={active} tooltip={ws.label}>
                      <NavLink to={ws.to}>
                        <Icon />
                        <span>{ws.label}</span>
                      </NavLink>
                    </SidebarMenuButton>
                    {active && (
                      <SidebarMenuSub>
                        {ws.pages.map((page) => {
                          const isPrefix = ws.pages.some(
                            (p) => p.to !== page.to && p.to.startsWith(`${page.to}/`),
                          );
                          const pageActive = isPrefix || page.to === ws.to
                            ? location.pathname === page.to
                            : location.pathname.startsWith(page.to);
                          return (
                            <SidebarMenuSubItem key={page.to}>
                              <SidebarMenuSubButton asChild isActive={pageActive}>
                                <NavLink to={page.to}>
                                  <span>{page.label}</span>
                                </NavLink>
                              </SidebarMenuSubButton>
                            </SidebarMenuSubItem>
                          );
                        })}
                      </SidebarMenuSub>
                    )}
                  </SidebarMenuItem>
                );
              })}
            </SidebarMenu>
          </SidebarGroupContent>
        </SidebarGroup>
      </SidebarContent>

      <SidebarFooter>
        <div className="flex items-center gap-2 px-2 py-1 group-data-[collapsible=icon]:justify-center">
          <span className={cn('led', gwOk ? 'text-profit bg-profit' : 'text-loss bg-loss')} />
          <span className="text-xs text-muted-foreground group-data-[collapsible=icon]:hidden">
            Gateway {gwOk ? '在线' : '离线'}
          </span>
          <Badge
            variant="outline"
            className={cn(
              'ml-auto font-mono text-[10px] group-data-[collapsible=icon]:hidden',
              liveEnabled ? 'border-loss/50 text-loss' : 'border-border text-muted-foreground',
            )}
          >
            {liveEnabled ? 'LIVE' : 'PAPER'}
          </Badge>
        </div>
      </SidebarFooter>
      <SidebarRail />
    </Sidebar>
  );
}

function findBreadcrumb(pathname: string): { ws?: string; page?: string } {
  if (pathname === '/') return { page: '仪表盘' };
  for (const ws of WORKSPACES) {
    for (const page of [...ws.pages].sort((a, b) => b.to.length - a.to.length)) {
      if (pathname.startsWith(page.to)) return { ws: ws.label, page: page.label };
    }
  }
  return {};
}

export default function Layout() {
  const location = useLocation();
  const crumb = findBreadcrumb(location.pathname);

  const [cmdOpen, setCmdOpen] = useState(false);
  const [light, setLight] = useState(() =>
    typeof window !== 'undefined' ? localStorage.getItem('theme') === 'light' : false,
  );
  const market = useMarketStore((s) => s.market);
  const setMarket = useMarketStore((s) => s.setMarket);

  const toggleTheme = () => {
    const next = !light;
    setLight(next);
    localStorage.setItem('theme', next ? 'light' : 'dark');
  };

  useEffect(() => {
    document.documentElement.classList.toggle('light', light);
    document.documentElement.classList.toggle('dark', !light);
  }, [light]);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === 'k') {
        e.preventDefault();
        setCmdOpen((v) => !v);
      }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, []);

  return (
    <SidebarProvider>
      <AppSidebar />
      <SidebarInset>
        <header className="flex h-14 shrink-0 items-center gap-2 border-b px-4">
          <SidebarTrigger className="-ml-1" />
          <Separator orientation="vertical" className="mr-1 !h-4" />
          <Breadcrumb>
            <BreadcrumbList>
              {crumb.ws && (
                <>
                  <BreadcrumbItem className="hidden md:block">
                    <span className="text-muted-foreground">{crumb.ws}</span>
                  </BreadcrumbItem>
                  <BreadcrumbSeparator className="hidden md:block">
                    <ChevronRight className="size-3.5" />
                  </BreadcrumbSeparator>
                </>
              )}
              <BreadcrumbItem>
                <BreadcrumbPage>{crumb.page || '—'}</BreadcrumbPage>
              </BreadcrumbItem>
            </BreadcrumbList>
          </Breadcrumb>

          <div className="ml-auto flex items-center gap-2">
            <div
              className="flex rounded-lg border bg-muted/40 p-0.5"
              title="页面上下文市场过滤"
            >
              {marketOptions.map((opt) => (
                <button
                  key={opt.value}
                  type="button"
                  onClick={() => setMarket(opt.value)}
                  className={cn(
                    'rounded-md px-2.5 py-1 text-xs transition-colors',
                    market === opt.value
                      ? 'bg-background text-foreground shadow-sm'
                      : 'text-muted-foreground hover:text-foreground',
                  )}
                >
                  {opt.label}
                </button>
              ))}
            </div>
            <Button
              variant="outline"
              size="sm"
              className="hidden md:flex text-muted-foreground"
              onClick={() => setCmdOpen(true)}
            >
              <Search className="size-3.5" />
              搜索
              <kbd className="pointer-events-none ml-1 inline-flex h-5 select-none items-center gap-0.5 rounded border bg-muted px-1.5 font-mono text-[10px] font-medium">
                ⌘K
              </kbd>
            </Button>
            <Button variant="ghost" size="icon-sm" onClick={toggleTheme} title="切换主题">
              {light ? <Moon className="size-4" /> : <Sun className="size-4" />}
            </Button>
          </div>
        </header>

        <main className="flex-1 overflow-y-auto min-h-0">
          <Outlet />
        </main>
      </SidebarInset>

      <CommandPalette open={cmdOpen} onClose={() => setCmdOpen(false)} />
    </SidebarProvider>
  );
}
