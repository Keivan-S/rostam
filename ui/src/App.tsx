import { Navigate, Route, Routes } from 'react-router-dom';
import { Layout } from './components/Layout';
import { Overview } from './pages/Overview';
import { Collections } from './pages/Collections';
import { SearchPage } from './pages/Search';
import { KVPage } from './pages/KV';
import { Admin } from './pages/Admin';

export default function App() {
  return (
    <Routes>
      <Route element={<Layout />}>
        <Route index element={<Overview />} />
        <Route path="collections" element={<Collections />} />
        <Route path="search" element={<SearchPage />} />
        <Route path="kv" element={<KVPage />} />
        <Route path="admin" element={<Admin />} />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Route>
    </Routes>
  );
}
