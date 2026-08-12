import type { ReactElement } from "react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, screen } from "@testing-library/react";
import { afterEach, expect, test, vi } from "vitest";
import AnomalyList from "./AnomalyList";
import InsightCard from "./InsightCard";

function withClient(ui: ReactElement) {
  const qc = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(<QueryClientProvider client={qc}>{ui}</QueryClientProvider>);
}

afterEach(() => vi.unstubAllGlobals());

test("AnomalyList renders flagged transactions with z-scores", async () => {
  vi.stubGlobal(
    "fetch",
    vi.fn().mockResolvedValue({
      ok: true,
      json: async () => ({
        anomalies: [
          {
            id: 1,
            date: "2026-06-15",
            merchant: "Sketchy Store",
            amount: 5000,
            category: "Groceries",
            description: "d",
            is_anomaly: true,
            z_score: 5.9,
            category_mean: 52.3,
            explanation: "This charge is 96x the category average.",
          },
        ],
      }),
    })
  );

  withClient(<AnomalyList />);

  expect(await screen.findByText("Sketchy Store")).toBeInTheDocument();
  expect(screen.getByText(/5\.9/)).toBeInTheDocument();
  expect(screen.getByText(/96x the category average/)).toBeInTheDocument();
});

test("InsightCard renders the latest stored insight", async () => {
  vi.stubGlobal(
    "fetch",
    vi.fn().mockResolvedValue({
      ok: true,
      json: async () => ({
        insights: [
          {
            id: 1,
            period: "2026-06",
            summary_text: "June spending was steady.",
            generated_at: "2026-07-01T00:00:00",
          },
        ],
      }),
    })
  );

  withClient(<InsightCard />);

  expect(await screen.findByText("June spending was steady.")).toBeInTheDocument();
  expect(screen.getByText(/2026-06/)).toBeInTheDocument();
});
