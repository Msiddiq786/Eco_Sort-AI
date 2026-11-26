import type { TrainLogEntry } from '@/types'
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from 'recharts'

interface Props {
  log: TrainLogEntry[]
}

export default function LossChart({ log }: Props) {
  return (
    <div className="card-eco overflow-hidden">
      <div className="section-header">
        <div>
          <h2 className="section-title">Training Convergence Curves</h2>
          <p className="text-xs text-slate-500 mt-0.5">
            Epoch-by-epoch training &amp; validation loss and accuracy metrics
          </p>
        </div>
        <span className="text-xs font-mono text-slate-500">{log.length} epochs</span>
      </div>

      <div className="p-5 grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div>
          <p className="text-xs font-semibold uppercase tracking-wider text-slate-500 mb-3">
            Cross-Entropy Loss
          </p>
          <ResponsiveContainer width="100%" height={240}>
            <LineChart data={log}>
              <CartesianGrid strokeDasharray="3 3" stroke="#E2E8F0" vertical={false} />
              <XAxis dataKey="epoch" tick={{ fontSize: 11, fill: '#64748B' }} />
              <YAxis tick={{ fontSize: 11, fill: '#64748B' }} domain={['auto', 'auto']} />
              <Tooltip
                contentStyle={{
                  borderRadius: '12px',
                  border: '1px solid #E2E8F0',
                  fontSize: '12px',
                }}
              />
              <Legend wrapperStyle={{ fontSize: '12px' }} />
              <Line
                type="monotone"
                dataKey="train_loss"
                stroke="#0EA5E9"
                name="Train Loss"
                dot={false}
                strokeWidth={2.5}
              />
              <Line
                type="monotone"
                dataKey="val_loss"
                stroke="#F43F5E"
                name="Val Loss"
                dot={false}
                strokeWidth={2.5}
              />
            </LineChart>
          </ResponsiveContainer>
        </div>

        <div>
          <p className="text-xs font-semibold uppercase tracking-wider text-slate-500 mb-3">
            Classification Accuracy
          </p>
          <ResponsiveContainer width="100%" height={240}>
            <LineChart data={log}>
              <CartesianGrid strokeDasharray="3 3" stroke="#E2E8F0" vertical={false} />
              <XAxis dataKey="epoch" tick={{ fontSize: 11, fill: '#64748B' }} />
              <YAxis
                tick={{ fontSize: 11, fill: '#64748B' }}
                domain={[0, 1]}
                tickFormatter={v => `${(v * 100).toFixed(0)}%`}
              />
              <Tooltip
                formatter={(v: number) => `${(v * 100).toFixed(2)}%`}
                contentStyle={{
                  borderRadius: '12px',
                  border: '1px solid #E2E8F0',
                  fontSize: '12px',
                }}
              />
              <Legend wrapperStyle={{ fontSize: '12px' }} />
              <Line
                type="monotone"
                dataKey="train_acc"
                stroke="#16A34A"
                name="Train Acc"
                dot={false}
                strokeWidth={2.5}
              />
              <Line
                type="monotone"
                dataKey="val_acc"
                stroke="#D4A54A"
                name="Val Acc"
                dot={false}
                strokeWidth={2.5}
              />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  )
}
