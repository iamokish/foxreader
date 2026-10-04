import { createContext } from "preact";
import { useState, useCallback, useContext, useEffect } from "preact/hooks";
import type { ComponentChildren } from "preact";
import { getInitialFoxConfig } from "../core/initialConfig";

export type Theme = "dark" | "light";

const STORAGE_KEY = "fox-reader-theme";

interface ThemeContextValue {
    theme: Theme;
    setTheme: (theme: Theme) => void;
}

const ThemeContext = createContext<ThemeContextValue>({
    theme: "dark",
    setTheme: () => {},
});

export function useTheme(): ThemeContextValue {
    return useContext(ThemeContext);
}

export function getInitialTheme(): Theme {
    try {
        const stored = localStorage.getItem(STORAGE_KEY);
        if (stored === "dark" || stored === "light") {
            return stored;
        }
    } catch {}

    const configTheme = getInitialFoxConfig().theme;
    if (configTheme === "dark" || configTheme === "light") {
        return configTheme;
    }

    return "dark";
}

export function ThemeProvider({ children }: { children: ComponentChildren }) {
    const [theme, setThemeState] = useState<Theme>(getInitialTheme);

    const setTheme = useCallback((newTheme: Theme) => {
        setThemeState(newTheme);
        try {
            localStorage.setItem(STORAGE_KEY, newTheme);
        } catch {}
    }, []);

    useEffect(() => {
        const html = document.documentElement;
        html.classList.remove("theme-dark", "theme-light");
        html.classList.add(`theme-${theme}`);
    }, [theme]);

    return <ThemeContext.Provider value={{ theme, setTheme }}>{children}</ThemeContext.Provider>;
}
