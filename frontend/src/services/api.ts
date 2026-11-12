import axios from 'axios';
import type {
  HealthResponse,
  ModelInfoResponse,
  PaginatedHistory,
  PredictResponse,
  StatsResponse,
  DetectResponse,
} from '@/types';

const api = axios.create({
  baseURL: '/api',
  timeout: 60_000,
});

// ── Health ────────────────────────────────────────────────────────────
export const getHealth = () =>
  api.get<HealthResponse>('/health').then(r => r.data);

export const getModelInfo = () =>
  api.get<ModelInfoResponse>('/model-info').then(r => r.data);

// ── Predict ───────────────────────────────────────────────────────────
export const predictImage = (file: File) => {
  const form = new FormData();
  form.append('file', file);
  return api.post<PredictResponse>('/predict', form, {
    headers: { 'Content-Type': 'multipart/form-data' },
  }).then(r => r.data);
};

// ── Detect (Multi-Object) ──────────────────────────────────────────────
export const detectObjects = (file: File) => {
  const form = new FormData();
  form.append('file', file);
  return api.post<DetectResponse>('/detect', form, {
    headers: { 'Content-Type': 'multipart/form-data' },
  }).then(r => r.data);
};

// ── Stats ─────────────────────────────────────────────────────────────
export const getStats = () =>
  api.get<StatsResponse>('/stats').then(r => r.data);

// ── History ───────────────────────────────────────────────────────────
export const getHistory = (params?: {
  page?: number;
  page_size?: number;
  category?: string;
  search?: string;
}) => api.get<PaginatedHistory>('/history', { params }).then(r => r.data);

export const deleteHistoryItem = (id: number) =>
  api.delete(`/history/${id}`);

export default api;
