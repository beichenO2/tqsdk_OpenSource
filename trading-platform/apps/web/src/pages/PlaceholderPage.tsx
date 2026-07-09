import { Link } from 'react-router-dom';
import { Construction } from 'lucide-react';
import Card from '@/components/Card';

interface PlaceholderPageProps {
  title: string;
  description: string;
  nextHint?: string;
  cta?: { to: string; label: string };
}

/** Lightweight placeholder for Web v2 pages not yet fully built. */
export default function PlaceholderPage({ title, description, nextHint, cta }: PlaceholderPageProps) {
  return (
    <div className="px-[3%] py-[2%] max-w-[56rem] mx-auto space-y-6">
      <div>
        <h1 className="text-xl font-semibold text-text-primary">{title}</h1>
        <p className="mt-1 text-sm text-text-secondary">{description}</p>
      </div>
      <Card>
        <div className="flex items-start gap-3">
          <Construction className="w-5 h-5 text-brand shrink-0 mt-0.5" />
          <div className="space-y-2 text-sm text-text-secondary">
            <p>本页为 Web v2 信息架构占位，功能将在后续 Phase 落地。</p>
            {nextHint && <p className="text-text-muted">{nextHint}</p>}
            {cta && (
              <Link to={cta.to} className="inline-flex text-brand hover:underline">
                {cta.label} →
              </Link>
            )}
          </div>
        </div>
      </Card>
    </div>
  );
}
