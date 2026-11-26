import axios from 'axios';
import type {
  AllRecommendationsResponse,
  BinRecommendation,
  EnsembleConfig,
  EnsembleEvaluation,
  FeedbackRequest,
  FeedbackResponse,
  FeedbackStatsResponse,
  HealthResponse,
  ModelInfoResponse,
  PaginatedDetectionHistory,
  PaginatedHistory,
  PredictResponse,
  StatsResponse,
  DetectResponse,
  XAIExplanation,
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
export const predictImage = (file: File, includeExplanation = false) => {
  const form = new FormData();
  form.append('file', file);
  const params = includeExplanation ? { include_explanation: true } : {};
  return api.post<PredictResponse>('/predict', form, {
    headers: { 'Content-Type': 'multipart/form-data' },
    params,
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

/**
 * Fetch all available prediction history records across pages.
 * Used by Analytics Dashboard to guarantee calculations are based on complete real records.
 */
export const getAllHistory = async (): Promise<import('@/types').HistoryItem[]> => {
  const firstPage = await getHistory({ page: 1, page_size: 100 });
  const allItems = [...firstPage.items];
  const total = firstPage.total;
  let currentPage = 2;
  while (allItems.length < total) {
    const nextPage = await getHistory({ page: currentPage, page_size: 100 });
    if (!nextPage.items.length) break;
    allItems.push(...nextPage.items);
    currentPage++;
  }
  return allItems;
};

export const deleteHistoryItem = (id: number) =>
  api.delete(`/history/${id}`);

export const getDetectionHistory = (params?: {
  page?: number;
  page_size?: number;
  search?: string;
}) => api.get<PaginatedDetectionHistory>('/history/detections', { params }).then(r => r.data);

export const deleteDetectionHistoryItem = (id: number) =>
  api.delete(`/history/detections/${id}`);

// ── Feedback (Active Learning) ────────────────────────────────────────
export const submitFeedback = (payload: FeedbackRequest) =>
  api.post<FeedbackResponse>('/feedback', payload).then(r => r.data);

export const getFeedbackStats = (limit = 50) =>
  api.get<FeedbackStatsResponse>('/feedback', { params: { limit } }).then(r => r.data);

// ── XAI ───────────────────────────────────────────────────────────────
export const getExplanation = (predictionId: number, includeHeatmap = true) =>
  api.get<XAIExplanation>(`/explain/${predictionId}`, {
    params: { include_heatmap: includeHeatmap },
  }).then(r => r.data);

export const explainImage = (file: File, includeHeatmap = true) => {
  const form = new FormData();
  form.append('file', file);
  return api.post<XAIExplanation>('/explain', form, {
    headers: { 'Content-Type': 'multipart/form-data' },
    params: { include_heatmap: includeHeatmap },
  }).then(r => r.data);
};

// ── Ensemble ──────────────────────────────────────────────────────────
export const getEnsembleConfig = () =>
  api.get<EnsembleConfig>('/ensemble/config').then(r => r.data);

export const getEnsembleEvaluation = () =>
  api.get<EnsembleEvaluation>('/ensemble/evaluation').then(r => r.data);

export const optimizeEnsemble = () =>
  api.post<{ status: string; message: string }>('/ensemble/optimize').then(r => r.data);

// ── Recommendations ───────────────────────────────────────────────────
export const getRecommendation = (category: string) =>
  api.get<BinRecommendation>(`/recommendations/${category}`).then(r => r.data);

export const getAllRecommendations = () =>
  api.get<AllRecommendationsResponse>('/recommendations').then(r => r.data);

export default api;
