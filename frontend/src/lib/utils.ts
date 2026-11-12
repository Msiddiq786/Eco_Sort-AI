import { clsx, type ClassValue } from 'clsx'
import { twMerge } from 'tailwind-merge'
import { WASTE_CLASSES, type WasteCategory } from '@/types'

export { WASTE_CLASSES, type WasteCategory }

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs))
}

/** Format a number as a percentage string */
export function formatPct(value: number, decimals = 1): string {
  return `${(value * 100).toFixed(decimals)}%`
}

/** Capitalise first letter */
export function capitalize(s: string): string {
  return s.charAt(0).toUpperCase() + s.slice(1)
}

/** Category → colour mapping (matches EcoSort UI theme) */
export const CATEGORY_COLORS: Record<string, string> = {
  cardboard: '#D4A54A',
  glass:     '#0EA5E9',
  metal:     '#64748B',
  paper:     '#10B981',
  plastic:   '#F97316',
  organic:   '#84CC16',
  trash:     '#6B7280',
}
