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
          forest:  '#14532D',
          emerald: '#16A34A',
          lime:    '#D9F99D',
          bg:      '#F7F9F5',
          mint:    '#ECFDF5',
          sage:    '#F0FDF4',
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
        waste: {
          cardboard: '#D4A54A',
          glass:     '#0EA5E9',
          metal:     '#64748B',
          paper:     '#10B981',
          plastic:   '#F97316',
          organic:   '#84CC16',
          trash:     '#6B7280',
        },
      },
      borderRadius: {
        '4xl': '2rem',
        lg: 'var(--radius)',
        md: 'calc(var(--radius) - 2px)',
        sm: 'calc(var(--radius) - 4px)',
      },
      boxShadow: {
        'eco-sm': '0 1px 3px rgba(20,83,45,0.06), 0 1px 2px rgba(0,0,0,0.04)',
        'eco':    '0 4px 16px rgba(20,83,45,0.08), 0 1px 4px rgba(0,0,0,0.04)',
        'eco-lg': '0 10px 40px rgba(20,83,45,0.12), 0 2px 8px rgba(0,0,0,0.06)',
        'card':   '0 0 0 1px rgba(20,83,45,0.08), 0 2px 8px rgba(0,0,0,0.04)',
      },
      backgroundImage: {
        'gradient-eco': 'linear-gradient(135deg, #14532D 0%, #16A34A 100%)',
        'gradient-eco-subtle': 'linear-gradient(135deg, #F0FDF4 0%, #ECFDF5 100%)',
        'gradient-eco-radial': 'radial-gradient(ellipse at top left, #ECFDF5 0%, #F7F9F5 60%)',
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
        'fade-in': {
          from: { opacity: '0', transform: 'translateY(4px)' },
          to: { opacity: '1', transform: 'translateY(0)' },
        },
        'slide-in-right': {
          from: { opacity: '0', transform: 'translateX(12px)' },
          to: { opacity: '1', transform: 'translateX(0)' },
        },
        'pulse-eco': {
          '0%, 100%': { opacity: '1' },
          '50%': { opacity: '0.6' },
        },
      },
      animation: {
        'accordion-down': 'accordion-down 0.2s ease-out',
        'accordion-up': 'accordion-up 0.2s ease-out',
        'fade-in': 'fade-in 0.3s ease-out',
        'slide-in-right': 'slide-in-right 0.3s ease-out',
        'pulse-eco': 'pulse-eco 2s ease-in-out infinite',
      },
    },
  },
  plugins: [require('tailwindcss-animate')],
}
