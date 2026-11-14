/**
 * Application shell: one header with the six sections (Overview, Replay lab, Results, Live
 * forecast, Live monitor, Method). Navigation is a URL hash so links can be shared without a router dependency;
 * switching section unmounts the previous one, which also discards any replay session state.
 *
 * PROTOTYPE REDESIGN: Results added as its own tab (it reads Role 4's committed reports, see
 * lib/results.ts); white masthead with an underline tab style; a footer with data credits.
 */
import { useEffect, useState } from "react";
import { LiveForecastPage } from "./pages/LiveForecast";
import { LiveMonitor } from "./pages/LiveMonitor";
import { Methodology, Overview } from "./pages/InfoPages";
import { ReplayLab } from "./pages/ReplayLab";
import { Results } from "./pages/Results";

export type Page = "overview" | "replay" | "results" | "forecast" | "monitor" | "method";
const PAGES: Array<[Page, string]> = [
  ["overview", "Overview"],
  ["replay", "Replay lab"],
  ["results", "Results"],
  // PROTOTYPE (2026-11-14): Live forecast hidden from the navigation until the trained model passes
  // Gate 2. The page and its route code stay; #forecast now falls back to Overview.
  ["monitor", "Live monitor"],
  ["method", "Method"],
];

const fromHash = (): Page => {
  const h = window.location.hash.replace("#", "") as Page;
  return PAGES.some(([p]) => p === h) ? h : "overview";
};

/** Wordmark glyph: a 3x3 block of 375 m cells with one cell lit, the unit FireGuard forecasts. */
function BrandMark() {
  const cells = [0, 1, 2].flatMap((r) => [0, 1, 2].map((c) => [r, c] as const));
  return (
    <svg className="brand-mark" viewBox="0 0 22 22" aria-hidden="true">
      {cells.map(([r, c]) => (
        <rect
          key={`${r}${c}`}
          x={c * 7.5}
          y={r * 7.5}
          width={6.5}
          height={6.5}
          rx={1}
          fill={r === 1 && c === 2 ? "#d9604e" : r === 0 && c === 2 ? "#f0bd57" : "#005838"}
          opacity={r === 1 && c === 2 ? 1 : r === 0 && c === 2 ? 1 : 0.22 + 0.1 * (r + c)}
        />
      ))}
    </svg>
  );
}

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
    window.scrollTo(0, 0);
  };

  return (
    <div className="app">
      <header className="masthead">
        <a
          className="brand"
          href="#overview"
          onClick={(e) => {
            e.preventDefault();
            go("overview");
          }}
        >
          <BrandMark />
          <span className="brand-name">FireGuard</span>
          <span className="brand-sub">Next-day wildfire activity, research prototype</span>
        </a>
        <nav className="tabs" aria-label="Sections">
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
        {page === "results" ? <Results /> : null}
        {page === "forecast" ? <LiveForecastPage /> : null}
        {page === "monitor" ? <LiveMonitor /> : null}
        {page === "method" ? <Methodology /> : null}
      </main>
      {page === "overview" || page === "results" || page === "method" ? (
        <footer className="footer">
          <p>
            Data: WildfireSpreadTS (Gerard et al., 2023), NASA FIRMS. Research prototype; not for operational
            use.
          </p>
          <p>FireGuard team project, 2026.</p>
        </footer>
      ) : null}
    </div>
  );
}
