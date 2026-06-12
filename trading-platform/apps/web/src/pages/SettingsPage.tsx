import Card from '@/components/Card';

export default function SettingsPage() {
  return (
    <div className="p-6 space-y-6">
      <h1 className="text-xl font-semibold text-text-primary">系统设置</h1>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        <Card title="交易账户">
          <div className="space-y-4">
            <div>
              <label className="block text-xs text-text-muted mb-1">TqSdk 账号</label>
              <input
                type="text"
                placeholder="请输入 TqSdk 账号"
                className="w-full bg-surface-tertiary border border-border rounded-lg px-3 py-2 text-sm text-text-primary focus:outline-none focus:border-brand"
              />
            </div>
            <div>
              <label className="block text-xs text-text-muted mb-1">期货公司</label>
              <select className="w-full bg-surface-tertiary border border-border rounded-lg px-3 py-2 text-sm text-text-primary focus:outline-none focus:border-brand">
                <option>快期模拟 (SimNow)</option>
                <option>中信期货</option>
                <option>海通期货</option>
                <option>国泰君安期货</option>
              </select>
            </div>
            <div>
              <label className="block text-xs text-text-muted mb-1">密码</label>
              <input
                type="password"
                placeholder="请输入密码"
                className="w-full bg-surface-tertiary border border-border rounded-lg px-3 py-2 text-sm text-text-primary focus:outline-none focus:border-brand"
              />
            </div>
            <button className="px-4 py-2 bg-brand text-white rounded-lg text-sm font-medium hover:bg-brand-dark transition-colors">
              保存配置
            </button>
          </div>
        </Card>

        <Card title="风控参数">
          <div className="space-y-4">
            <div>
              <label className="block text-xs text-text-muted mb-1">风险度预警线 (%)</label>
              <input
                type="number"
                defaultValue={30}
                className="w-full bg-surface-tertiary border border-border rounded-lg px-3 py-2 text-sm text-text-primary tabular-nums focus:outline-none focus:border-brand"
              />
            </div>
            <div>
              <label className="block text-xs text-text-muted mb-1">最大单品种持仓占比 (%)</label>
              <input
                type="number"
                defaultValue={50}
                className="w-full bg-surface-tertiary border border-border rounded-lg px-3 py-2 text-sm text-text-primary tabular-nums focus:outline-none focus:border-brand"
              />
            </div>
            <div>
              <label className="block text-xs text-text-muted mb-1">单日最大亏损 (元)</label>
              <input
                type="number"
                defaultValue={50000}
                className="w-full bg-surface-tertiary border border-border rounded-lg px-3 py-2 text-sm text-text-primary tabular-nums focus:outline-none focus:border-brand"
              />
            </div>
            <button className="px-4 py-2 bg-brand text-white rounded-lg text-sm font-medium hover:bg-brand-dark transition-colors">
              保存参数
            </button>
          </div>
        </Card>

        <Card title="系统信息">
          <div className="space-y-2 text-sm">
            <div className="flex justify-between">
              <span className="text-text-muted">应用</span>
              <span className="text-text-primary font-mono">PolarTrade v0.1.0</span>
            </div>
            <div className="flex justify-between">
              <span className="text-text-muted">API 地址</span>
              <span className="text-text-primary font-mono">localhost:8000</span>
            </div>
            <div className="flex justify-between">
              <span className="text-text-muted">WebSocket</span>
              <span className="text-text-primary font-mono">ws://localhost:8000/ws</span>
            </div>
            <div className="flex justify-between">
              <span className="text-text-muted">数据源</span>
              <span className="text-text-primary">Mock 数据（待接入后端）</span>
            </div>
          </div>
        </Card>

        <Card title="通知设置">
          <div className="space-y-3">
            {[
              { label: '风控预警通知', defaultChecked: true },
              { label: '策略异常通知', defaultChecked: true },
              { label: '成交回报通知', defaultChecked: false },
              { label: '每日盈亏报告', defaultChecked: false },
            ].map(item => (
              <label key={item.label} className="flex items-center justify-between cursor-pointer">
                <span className="text-sm text-text-secondary">{item.label}</span>
                <input
                  type="checkbox"
                  defaultChecked={item.defaultChecked}
                  className="w-4 h-4 rounded border-border text-brand focus:ring-brand bg-surface-tertiary"
                />
              </label>
            ))}
          </div>
        </Card>
      </div>
    </div>
  );
}
