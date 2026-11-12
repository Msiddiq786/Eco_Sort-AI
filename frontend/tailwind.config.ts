/** @type {import('tailwindcss').Config} */
export default {
  darkMode: ['class'],
  content: ['./index.html', './src/**/*.{ts,tsx}'],
  theme: {
    extend: {
      fontFamily: {
        sans: ['Inter', 'system-ui', 'sans-serif'],
        heading: ['Manrope', 'system-ui', 'sans-serif'],
      },
      colors: {
        border: 'hsl(var(--border))',
        input: 'hsl(var(--input))',
        ring: 'hsl(var(--ring))',
        background: '#F7F9F5',
        foreground: 'hsl(var(--foreground))',
        card: {
          DEFAULT: '#FFFFFF',
          foreground: 'hsl(var(--card-foreground))',
        },
        ecosort: {
          forest: '#14532D',
          emerald: '#16A34A',
          lime: '#D9F99D',
          bg: '#F7F9F5',
        },
        primary: {
          DEFAULT: '#16A34A',
          dark: '#14532D',
          foreground: '#FFFFFF',
        },
        secondary: {
          DEFAULT: '#D9F99D',
          foreground: '#14532D',
        },
        destructive: {
          DEFAULT: 'hsl(var(--destructive))',
          foreground: 'hsl(var(--destructive-foreground))',
        },
        muted: {
          DEFAULT: '#F1F5F0',
          foreground: '#64748B',
        },
        accent: {
          DEFAULT: '#E8F5E9',
          foreground: '#14532D',
        },
        // Waste category colors
        waste: {
          cardboard: '#D4A54A',
          glass:     '#0EA5E9',
          metal:     '#64748B',
          paper:     '#10B981',
          plastic:   '#F97316',
          trash:     '#6B7280',
        },
      },
      borderRadius: {
        lg: 'var(--radius)',
        md: 'calc(var(--radius) - 2px)',
        sm: 'calc(var(--radius) - 4px)',
      },
      keyframes: {
        'accordion-down': {
          from: { height: '0' },
          to: { height: 'var(--radix-accordion-content-height)' },
        },
        'accordion-up': {
          from: { height: 'var(--radix-accordion-content-height)' },
          to: { height: '0' },
        },
      },
      animation: {
        'accordion-down': 'accordion-down 0.2s ease-out',
        'accordion-up': 'accordion-up 0.2s ease-out',
      },
    },
  },
  plugins: [require('tailwindcss-animate')],
}
