/**
 * Application shell: one header with the four sections from the plan (Overview, Live Monitor,
 * Forecast & Replay Lab, Methodology). Navigation is a URL hash so links can be shared without a
 * router dependency; switching section unmounts the previous one, which also discards any replay
 * session state.
 */
import { useEffect, useState } from "react";
import { LiveMonitor } from "./pages/LiveMonitor";
import { Methodology, Overview } from "./pages/InfoPages";
import { ReplayLab } from "./pages/ReplayLab";

type Page = "overview" | "replay" | "monitor" | "method";
const PAGES: Array<[Page, string]> = [
  ["overview", "Overview"],
  ["replay", "Replay lab"],
  ["monitor", "Live monitor"],
  ["method", "Method"],
];

const fromHash = (): Page => {
  const h = window.location.hash.replace("#", "") as Page;
  return PAGES.some(([p]) => p === h) ? h : "overview";
};

export default function App() {
  const [page, setPage] = useState<Page>(fromHash);
  useEffect(() => {
    const onHash = () => setPage(fromHash());
    window.addEventListener("hashchange", onHash);
    return () => window.removeEventListener("hashchange", onHash);
  }, []);
  const go = (p: Page) => {
    window.location.hash = p;
    setPage(p);
  };

  return (
    <div className="app">
      <header className="masthead">
        <a className="brand" href="#overview" onClick={() => go("overview")}>
          FireGuard
          <span className="brand-sub">next-day wildfire activity, research prototype</span>
        </a>
        <nav aria-label="Sections">
          {PAGES.map(([p, label]) => (
            <a
              key={p}
              href={`#${p}`}
              aria-current={page === p ? "page" : undefined}
              onClick={(e) => {
                e.preventDefault();
                go(p);
              }}
            >
              {label}
            </a>
          ))}
        </nav>
      </header>
      <main id="main">
        {page === "overview" ? <Overview go={go} /> : null}
        {page === "replay" ? <ReplayLab /> : null}
        {page === "monitor" ? <LiveMonitor /> : null}
        {page === "method" ? <Methodology /> : null}
      </main>
    </div>
  );
}
