import type { TrainLogEntry } from '@/types'
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid,
  Tooltip, Legend, ResponsiveContainer,
} from 'recharts'

interface Props { log: TrainLogEntry[] }

export default function LossChart({ log }: Props) {
  return (
    <div className="rounded-lg border border-border bg-card p-5 space-y-6">
      <h2 className="font-semibold text-sm">Training History</h2>

      <div>
        <p className="text-xs text-muted-foreground mb-2">Loss</p>
        <ResponsiveContainer width="100%" height={200}>
          <LineChart data={log}>
            <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" />
            <XAxis dataKey="epoch" tick={{ fontSize: 11 }} />
            <YAxis tick={{ fontSize: 11 }} domain={['auto', 'auto']} />
            <Tooltip />
            <Legend />
            <Line type="monotone" dataKey="train_loss" stroke="#5B9BD5" name="Train Loss" dot={false} strokeWidth={2} />
            <Line type="monotone" dataKey="val_loss"   stroke="#E06C75" name="Val Loss"   dot={false} strokeWidth={2} />
          </LineChart>
        </ResponsiveContainer>
      </div>

      <div>
        <p className="text-xs text-muted-foreground mb-2">Accuracy</p>
        <ResponsiveContainer width="100%" height={200}>
          <LineChart data={log}>
            <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" />
            <XAxis dataKey="epoch" tick={{ fontSize: 11 }} />
            <YAxis tick={{ fontSize: 11 }} domain={[0, 1]} tickFormatter={v => `${(v * 100).toFixed(0)}%`} />
            <Tooltip formatter={(v: number) => `${(v * 100).toFixed(2)}%`} />
            <Legend />
            <Line type="monotone" dataKey="train_acc" stroke="#98C379" name="Train Acc" dot={false} strokeWidth={2} />
            <Line type="monotone" dataKey="val_acc"   stroke="#D4A54A" name="Val Acc"   dot={false} strokeWidth={2} />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  )
}
