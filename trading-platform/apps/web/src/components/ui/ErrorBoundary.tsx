import { Component, type ReactNode, type ErrorInfo } from 'react';
import { AlertTriangle, RotateCcw } from 'lucide-react';
import { Button } from './Button';

interface Props {
  children: ReactNode;
  fallback?: ReactNode;
}

interface State {
  error: Error | null;
}

export class ErrorBoundary extends Component<Props, State> {
  state: State = { error: null };

  static getDerivedStateFromError(error: Error): State {
    return { error };
  }

  componentDidCatch(error: Error, info: ErrorInfo) {
    console.error('[ErrorBoundary]', error, info.componentStack);
  }

  handleReset = () => {
    this.setState({ error: null });
  };

  render() {
    if (this.state.error) {
      if (this.props.fallback) return this.props.fallback;
      return (
        <div className="flex flex-col items-center justify-center gap-4 p-12 text-center">
          <div className="rounded-full bg-loss/15 p-3">
            <AlertTriangle className="h-6 w-6 text-loss" />
          </div>
          <div>
            <h3 className="text-base font-semibold text-text-primary">页面出错了</h3>
            <p className="mt-1 text-sm text-text-muted max-w-md">
              {this.state.error.message || '发生了未知错误'}
            </p>
          </div>
          <Button variant="secondary" onClick={this.handleReset}>
            <RotateCcw className="h-4 w-4" />
            重试
          </Button>
        </div>
      );
    }
    return this.props.children;
  }
}
