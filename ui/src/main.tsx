import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
import { HashRouter } from 'react-router-dom';
import App from './App';
import { ThemeProvider } from './context/ThemeContext';
import { ApiKeyProvider } from './context/ApiKeyContext';
import { SettingsProvider } from './context/SettingsContext';
import './index.css';

// HashRouter keeps client-side routing working under the /dashboard/ prefix
// without any server-side rewrite — the maintainer only has to serve index.html
// at one path.
createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <ThemeProvider>
      <SettingsProvider>
        <ApiKeyProvider>
          <HashRouter>
            <App />
          </HashRouter>
        </ApiKeyProvider>
      </SettingsProvider>
    </ThemeProvider>
  </StrictMode>,
);
