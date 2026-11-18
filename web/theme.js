// Light/dark theme switch. Follows the device theme until the user picks one;
// picking the device's own theme again goes back to following it.
// Load it in <head> without defer so the saved theme applies before first paint,
// and put <button class="theme-toggle" data-theme-toggle></button> where the switch should be.
(() => {
  const KEY = 'theme';
  const root = document.documentElement;
  const media = matchMedia('(prefers-color-scheme: dark)');
  const read = () => { try { return localStorage.getItem(KEY); } catch { return null; } };
  const write = v => { try { v ? localStorage.setItem(KEY, v) : localStorage.removeItem(KEY); } catch {} };
  const device = () => (media.matches ? 'dark' : 'light');
  const current = () => root.dataset.theme || device();

  const saved = read();
  if (saved === 'light' || saved === 'dark') root.dataset.theme = saved;

  const SUN = '<svg viewBox="0 0 24 24" width="18" height="18" aria-hidden="true" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><circle cx="12" cy="12" r="4.2"/><path d="M12 2.5v2.2M12 19.3v2.2M2.5 12h2.2M19.3 12h2.2M5.3 5.3l1.6 1.6M17.1 17.1l1.6 1.6M5.3 18.7l1.6-1.6M17.1 6.9l1.6-1.6"/></svg>';
  const MOON = '<svg viewBox="0 0 24 24" width="18" height="18" aria-hidden="true" fill="none" stroke="currentColor" stroke-width="2" stroke-linejoin="round"><path d="M20.5 14.2A8.5 8.5 0 1 1 9.8 3.5a6.8 6.8 0 0 0 10.7 10.7z"/></svg>';

  const style = document.createElement('style');
  style.textContent = `.theme-toggle{flex:none;display:inline-grid;place-items:center;width:36px;height:36px;padding:0;margin-left:auto;border-radius:10px;cursor:pointer;
    border:1px solid var(--line,var(--border,#8884));background:var(--panel,var(--surface,transparent));color:var(--ink,var(--text,inherit))}
    .theme-toggle:hover{border-color:var(--accent,currentColor)}
    .theme-toggle:focus-visible{outline:2px solid var(--accent,currentColor);outline-offset:2px}`;
  document.head.appendChild(style);

  const paint = () => {
    const t = current(), next = t === 'dark' ? 'light' : 'dark';
    const label = `Switch to ${next} theme` + (root.dataset.theme ? '' : ' (now following your device)');
    document.querySelectorAll('[data-theme-toggle]').forEach(b => {
      b.innerHTML = t === 'dark' ? SUN : MOON;
      b.title = label;
      b.setAttribute('aria-label', label);
    });
  };
  const changed = () => { paint(); dispatchEvent(new CustomEvent('themechange', { detail: { theme: current() } })); };

  const toggle = () => {
    const next = current() === 'dark' ? 'light' : 'dark';
    if (next === device()) { delete root.dataset.theme; write(null); }
    else { root.dataset.theme = next; write(next); }
    changed();
  };

  media.addEventListener('change', () => { if (!root.dataset.theme) changed(); });
  addEventListener('storage', e => {
    if (e.key !== KEY) return;
    if (e.newValue) root.dataset.theme = e.newValue; else delete root.dataset.theme;
    changed();
  });
  const bind = () => {
    document.addEventListener('click', e => { if (e.target.closest('[data-theme-toggle]')) toggle(); });
    paint();
  };
  document.readyState === 'loading' ? document.addEventListener('DOMContentLoaded', bind) : bind();
  window.themeToggle = { toggle, current };
})();
