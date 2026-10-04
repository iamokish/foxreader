import { ThemeProvider } from "./themes";
import { LayoutProvider, ActiveLayout } from "./layouts";
import { WorkspaceProvider } from "./workspace";

export function App() {
    return (
        <ThemeProvider>
            <LayoutProvider>
                <WorkspaceProvider>
                    <ActiveLayout />
                </WorkspaceProvider>
            </LayoutProvider>
        </ThemeProvider>
    );
}
