/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        background: "#f8f9ff",
        surface: "#ffffff",
        "surface-low": "#eff4ff",
        "surface-mid": "#e5eeff",
        "surface-high": "#dce9ff",
        primary: "#006591",
        "primary-bright": "#0ea5e9",
        secondary: "#006b5f",
        "secondary-bright": "#14b8a6",
        outline: "#bec8d2",
        text: "#0b1c30",
        muted: "#3e4850",
        danger: "#ba1a1a",
        warning: "#de8712"
      },
      fontFamily: {
        heading: ["Manrope", "sans-serif"],
        sans: ["Inter", "sans-serif"]
      },
      spacing: {
        sidebar: "260px"
      },
      borderRadius: {
        panel: "8px"
      },
      boxShadow: {
        soft: "0 18px 60px rgba(15, 38, 70, 0.08)",
        glow: "0 12px 30px rgba(14, 165, 233, 0.25)"
      }
    }
  },
  plugins: []
};
