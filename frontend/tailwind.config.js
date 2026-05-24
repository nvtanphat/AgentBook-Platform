/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  darkMode: 'class',
  theme: {
    extend: {
      colors: {
        background: "var(--c-background)",
        surface: "var(--c-surface)",
        "surface-low": "var(--c-surface-low)",
        "surface-mid": "var(--c-surface-mid)",
        "surface-high": "var(--c-surface-high)",
        primary: "var(--c-primary)",
        "primary-bright": "var(--c-primary-bright)",
        secondary: "var(--c-secondary)",
        "secondary-bright": "var(--c-secondary-bright)",
        outline: "var(--c-outline)",
        text: "var(--c-text)",
        muted: "var(--c-muted)",
        danger: "var(--c-danger)",
        warning: "var(--c-warning)"
      },
      fontFamily: {
        heading: ["Manrope", "sans-serif"],
        sans: ["Inter", "sans-serif"]
      },
      spacing: {
        sidebar: "260px"
      },
      borderRadius: {
        panel: "12px"
      },
      boxShadow: {
        soft: "0 18px 60px rgba(15, 38, 70, 0.06)",
        glow: "0 12px 30px rgba(14, 165, 233, 0.2)",
        card: "0 1px 3px rgba(15, 38, 70, 0.04), 0 4px 12px rgba(15, 38, 70, 0.02)"
      }
    }
  },
  plugins: []
};
