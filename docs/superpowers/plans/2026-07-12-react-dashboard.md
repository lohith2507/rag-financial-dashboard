# React Dashboard + Chat UI Implementation Plan (Plan 4 of 4)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A polished single-page dashboard — spend charts, anomaly alerts, AI insight cards, and a RAG chat panel that surfaces which tools the agent used — plus the project README.

**Architecture:** Vite + React 18 + TypeScript SPA. TanStack Query fetches from the Plan-3 endpoints through a Vite dev proxy (no CORS config needed). Tremor supplies finance-grade charts on Tailwind. A thin typed API client isolates all fetch logic; components stay presentational + one `useQuery`/`useMutation` each.

**Tech Stack:** Vite 5, React 18.3, TypeScript, TailwindCSS 3.4, @tremor/react 3.x, TanStack Query v5, Vitest + Testing Library (fetch stubbed — tests never hit a server).

**Prerequisite:** Plans 1–3 complete; backend runs at `http://localhost:8000` (`uvicorn app.main:app`).

## Global Constraints

- **Pin the UI stack:** `react@18.3.1`, `react-dom@18.3.1`, `@tremor/react@^3.18`, `tailwindcss@^3.4` (NOT Tailwind v4 — Tremor 3.x requires v3), `@tanstack/react-query@^5`.
- All backend access goes through `frontend/src/api/client.ts` — components never call `fetch` directly.
- Backend paths are proxied by Vite dev server: `/chat`, `/stats`, `/anomalies`, `/insights`, `/transactions`, `/health` → `http://localhost:8000`.
- Tests stub `fetch` (`vi.stubGlobal`) — no network, no running backend needed.
- Every task ends with passing tests (`npm test -- --run`) and a commit.

---

### Task 1: Frontend scaffold (Vite + Tailwind + Tremor + Query + Vitest)

**Files:**
- Create: `frontend/` via scaffolder, then modify: `frontend/package.json` (pins), `frontend/vite.config.ts`, `frontend/tailwind.config.js`, `frontend/postcss.config.js`, `frontend/src/index.css`, `frontend/src/main.tsx`, `frontend/src/App.tsx`, `frontend/src/setupTests.ts`
- Test: `frontend/src/App.test.tsx`

**Interfaces:**
- Produces: running dev server (`npm run dev`) proxying to the backend; `npm test -- --run` green; `App` renders the page shell with the heading "AI Financial Insights".

- [ ] **Step 1: Scaffold and install pinned deps**

Run from repo root:
```bash
npm create vite@latest frontend -- --template react-ts
cd frontend
npm install react@18.3.1 react-dom@18.3.1
npm install @tremor/react@^3.18 @tanstack/react-query@^5
npm install -D tailwindcss@^3.4 postcss autoprefixer @types/react@^18 @types/react-dom@^18
npm install -D vitest jsdom @testing-library/react @testing-library/jest-dom @testing-library/user-event
npx tailwindcss init -p
```

- [ ] **Step 2: Configure Vite (proxy + vitest)**

`frontend/vite.config.ts`:
```ts
/// <reference types="vitest" />
import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

const backend = "http://localhost:8000";
const proxiedPaths = ["/chat", "/stats", "/anomalies", "/insights", "/transactions", "/health"];

export default defineConfig({
  plugins: [react()],
  server: {
    proxy: Object.fromEntries(proxiedPaths.map((p) => [p, backend])),
  },
  test: {
    environment: "jsdom",
    globals: true,
    setupFiles: "./src/setupTests.ts",
  },
});
```

`frontend/src/setupTests.ts`:
```ts
import "@testing-library/jest-dom";
```

Add to `frontend/package.json` scripts: `"test": "vitest"`.

- [ ] **Step 3: Configure Tailwind for Tremor**

`frontend/tailwind.config.js`:
```js
/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{ts,tsx}",
    "./node_modules/@tremor/**/*.{js,ts,jsx,tsx}",
  ],
  theme: { extend: {} },
  plugins: [],
};
```

`frontend/src/index.css` (replace contents):
```css
@tailwind base;
@tailwind components;
@tailwind utilities;
```

- [ ] **Step 4: Write the failing shell test**

`frontend/src/App.test.tsx`:
```tsx
import { render, screen } from "@testing-library/react";
import App from "./App";

test("renders the app heading", () => {
  render(<App />);
  expect(screen.getByRole("heading", { name: /AI Financial Insights/i })).toBeInTheDocument();
});
```

Run: `npm test -- --run` — Expected: FAIL (default Vite App has no such heading).

- [ ] **Step 5: Implement the shell**

`frontend/src/App.tsx` (replace):
```tsx
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";

const queryClient = new QueryClient();

export default function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <main className="min-h-screen bg-slate-950 p-6 text-slate-100">
        <h1 className="text-2xl font-semibold">AI Financial Insights</h1>
        <p className="mt-1 text-sm text-slate-400">
          RAG-grounded answers, anomaly alerts, and plain-English summaries of your money.
        </p>
      </main>
    </QueryClientProvider>
  );
}
```

`frontend/src/main.tsx` should render `<App />` inside `React.StrictMode` and import `./index.css` (Vite default already does — verify).

- [ ] **Step 6: Verify test passes and dev server boots**

Run: `npm test -- --run` — Expected: 1 passed.
Run: `npm run dev` briefly — Expected: page shows the heading (Ctrl+C after).

- [ ] **Step 7: Commit**

```bash
git add frontend
git commit -m "feat: scaffold react dashboard shell (vite+tailwind+tremor+query)"
```

---

### Task 2: Typed API client

**Files:**
- Create: `frontend/src/api/client.ts`
- Test: `frontend/src/api/client.test.ts`

**Interfaces:**
- Consumes: Plan 2/3 endpoint shapes (verbatim).
- Produces:
```ts
export interface CategoryTotal { category: string; total: number }
export interface MonthlyPoint { period: string; spend: number; income: number }
export interface Txn { id: number; date: string; merchant: string; amount: number;
                       category: string; description: string; is_anomaly: boolean }
export interface Anomaly extends Txn { z_score: number; category_mean: number; explanation?: string }
export interface Insight { id: number; period: string; summary_text: string; generated_at: string | null }
export interface ChatReply { answer: string; tools_used: { tool: string; args: Record<string, unknown> }[] }

getByCategory(): Promise<CategoryTotal[]>
getMonthly(): Promise<MonthlyPoint[]>
getAnomalies(explain?: boolean): Promise<Anomaly[]>
getInsights(): Promise<Insight[]>
generateInsight(period: string): Promise<Insight>
getTransactions(limit?: number): Promise<Txn[]>
sendChat(message: string, sessionId?: string): Promise<ChatReply>
```

- [ ] **Step 1: Write the failing tests**

`frontend/src/api/client.test.ts`:
```ts
import { afterEach, expect, test, vi } from "vitest";
import { getAnomalies, getByCategory, sendChat } from "./client";

function stubFetch(body: unknown, ok = true) {
  const fn = vi.fn().mockResolvedValue({ ok, status: ok ? 200 : 500, json: async () => body });
  vi.stubGlobal("fetch", fn);
  return fn;
}

afterEach(() => vi.unstubAllGlobals());

test("getByCategory unwraps the payload", async () => {
  stubFetch({ by_category: [{ category: "Groceries", total: 80 }] });
  expect(await getByCategory()).toEqual([{ category: "Groceries", total: 80 }]);
});

test("getAnomalies passes explain flag", async () => {
  const fn = stubFetch({ anomalies: [] });
  await getAnomalies(true);
  expect(fn).toHaveBeenCalledWith("/anomalies?explain=true");
});

test("sendChat posts message and returns reply", async () => {
  const fn = stubFetch({ answer: "You spent $80.", tools_used: [] });
  const reply = await sendChat("how much?", "s1");
  expect(reply.answer).toBe("You spent $80.");
  const [url, init] = fn.mock.calls[0];
  expect(url).toBe("/chat");
  expect(JSON.parse((init as RequestInit).body as string)).toEqual({
    message: "how much?", session_id: "s1",
  });
});

test("errors throw with status", async () => {
  stubFetch({}, false);
  await expect(getByCategory()).rejects.toThrow(/500/);
});
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `npm test -- --run src/api` — Expected: FAIL (module doesn't exist).

- [ ] **Step 3: Implement**

`frontend/src/api/client.ts`:
```ts
export interface CategoryTotal { category: string; total: number }
export interface MonthlyPoint { period: string; spend: number; income: number }
export interface Txn {
  id: number; date: string; merchant: string; amount: number;
  category: string; description: string; is_anomaly: boolean;
}
export interface Anomaly extends Txn { z_score: number; category_mean: number; explanation?: string }
export interface Insight { id: number; period: string; summary_text: string; generated_at: string | null }
export interface ChatReply {
  answer: string;
  tools_used: { tool: string; args: Record<string, unknown> }[];
}

async function get<T>(url: string): Promise<T> {
  const resp = await fetch(url);
  if (!resp.ok) throw new Error(`GET ${url} failed: ${resp.status}`);
  return resp.json() as Promise<T>;
}

async function post<T>(url: string, body: unknown): Promise<T> {
  const resp = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!resp.ok) throw new Error(`POST ${url} failed: ${resp.status}`);
  return resp.json() as Promise<T>;
}

export const getByCategory = () =>
  get<{ by_category: CategoryTotal[] }>("/stats/by-category").then((d) => d.by_category);

export const getMonthly = () =>
  get<{ monthly: MonthlyPoint[] }>("/stats/monthly").then((d) => d.monthly);

export const getAnomalies = (explain = false) =>
  get<{ anomalies: Anomaly[] }>(`/anomalies?explain=${explain}`).then((d) => d.anomalies);

export const getInsights = () =>
  get<{ insights: Insight[] }>("/insights").then((d) => d.insights);

export const generateInsight = (period: string) =>
  post<Insight>(`/insights/${period}/generate`, {});

export const getTransactions = (limit = 50) =>
  get<{ transactions: Txn[] }>(`/transactions?limit=${limit}`).then((d) => d.transactions);

export const sendChat = (message: string, sessionId = "default") =>
  post<ChatReply>("/chat", { message, session_id: sessionId });
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `npm test -- --run src/api` — Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add frontend/src/api
git commit -m "feat: add typed api client"
```

---

### Task 3: Dashboard widgets (charts, anomalies, insights)

**Files:**
- Create: `frontend/src/components/SpendByCategory.tsx`
- Create: `frontend/src/components/MonthlyTrend.tsx`
- Create: `frontend/src/components/AnomalyList.tsx`
- Create: `frontend/src/components/InsightCard.tsx`
- Test: `frontend/src/components/widgets.test.tsx`

**Interfaces:**
- Consumes: API client functions; TanStack Query.
- Produces: four self-contained widgets, each `export default function Widget()` with its own `useQuery` (keys: `["by-category"]`, `["monthly"]`, `["anomalies"]`, `["insights"]`). `InsightCard` also has a `useMutation` calling `generateInsight` for the latest month and invalidating `["insights"]`.

- [ ] **Step 1: Write the failing tests** (test the DOM-verifiable widgets; chart internals are Tremor's concern)

`frontend/src/components/widgets.test.tsx`:
```tsx
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, screen } from "@testing-library/react";
import { afterEach, expect, test, vi } from "vitest";
import AnomalyList from "./AnomalyList";
import InsightCard from "./InsightCard";

function withClient(ui: React.ReactElement) {
  const qc = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(<QueryClientProvider client={qc}>{ui}</QueryClientProvider>);
}

afterEach(() => vi.unstubAllGlobals());

test("AnomalyList renders flagged transactions with z-scores", async () => {
  vi.stubGlobal("fetch", vi.fn().mockResolvedValue({
    ok: true,
    json: async () => ({
      anomalies: [{ id: 1, date: "2026-06-15", merchant: "Sketchy Store", amount: 5000,
                    category: "Groceries", description: "d", is_anomaly: true,
                    z_score: 5.9, category_mean: 52.3,
                    explanation: "This charge is 96x the category average." }],
    }),
  }));

  withClient(<AnomalyList />);

  expect(await screen.findByText("Sketchy Store")).toBeInTheDocument();
  expect(screen.getByText(/5\.9/)).toBeInTheDocument();
  expect(screen.getByText(/96x the category average/)).toBeInTheDocument();
});

test("InsightCard renders the latest stored insight", async () => {
  vi.stubGlobal("fetch", vi.fn().mockResolvedValue({
    ok: true,
    json: async () => ({
      insights: [{ id: 1, period: "2026-06", summary_text: "June spending was steady.",
                   generated_at: "2026-07-01T00:00:00" }],
    }),
  }));

  withClient(<InsightCard />);

  expect(await screen.findByText("June spending was steady.")).toBeInTheDocument();
  expect(screen.getByText(/2026-06/)).toBeInTheDocument();
});
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `npm test -- --run src/components` — Expected: FAIL (modules don't exist).

- [ ] **Step 3: Implement the four widgets**

`frontend/src/components/SpendByCategory.tsx`:
```tsx
import { useQuery } from "@tanstack/react-query";
import { Card, DonutChart, Title } from "@tremor/react";
import { getByCategory } from "../api/client";

export default function SpendByCategory() {
  const { data = [] } = useQuery({ queryKey: ["by-category"], queryFn: () => getByCategory() });
  return (
    <Card>
      <Title>Spend by Category</Title>
      <DonutChart className="mt-4 h-52" data={data} category="total" index="category"
                  valueFormatter={(v) => `$${v.toLocaleString()}`} />
    </Card>
  );
}
```

`frontend/src/components/MonthlyTrend.tsx`:
```tsx
import { useQuery } from "@tanstack/react-query";
import { BarChart, Card, Title } from "@tremor/react";
import { getMonthly } from "../api/client";

export default function MonthlyTrend() {
  const { data = [] } = useQuery({ queryKey: ["monthly"], queryFn: getMonthly });
  return (
    <Card>
      <Title>Spend vs Income by Month</Title>
      <BarChart className="mt-4 h-52" data={data} index="period"
                categories={["spend", "income"]}
                valueFormatter={(v) => `$${v.toLocaleString()}`} />
    </Card>
  );
}
```

`frontend/src/components/AnomalyList.tsx`:
```tsx
import { useQuery } from "@tanstack/react-query";
import { Badge, Card, Title } from "@tremor/react";
import { getAnomalies } from "../api/client";

export default function AnomalyList() {
  const { data = [] } = useQuery({
    queryKey: ["anomalies"],
    queryFn: () => getAnomalies(true),
  });
  return (
    <Card>
      <Title>Anomaly Alerts</Title>
      {data.length === 0 && <p className="mt-3 text-sm text-tremor-content">No anomalies detected.</p>}
      <ul className="mt-3 space-y-3">
        {data.map((a) => (
          <li key={a.id} className="rounded border border-red-900/40 p-3">
            <div className="flex items-center justify-between">
              <span className="font-medium">{a.merchant}</span>
              <Badge color="red">z = {a.z_score}</Badge>
            </div>
            <div className="text-sm text-tremor-content">
              {a.date} · {a.category} · ${a.amount.toLocaleString()}
            </div>
            {a.explanation && <p className="mt-1 text-sm">{a.explanation}</p>}
          </li>
        ))}
      </ul>
    </Card>
  );
}
```

`frontend/src/components/InsightCard.tsx`:
```tsx
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Button, Card, Title } from "@tremor/react";
import { generateInsight, getInsights } from "../api/client";

function latestPeriod(): string {
  const now = new Date();
  return `${now.getFullYear()}-${String(now.getMonth() + 1).padStart(2, "0")}`;
}

export default function InsightCard() {
  const qc = useQueryClient();
  const { data = [] } = useQuery({ queryKey: ["insights"], queryFn: getInsights });
  const generate = useMutation({
    mutationFn: () => generateInsight(latestPeriod()),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["insights"] }),
  });
  const latest = data[0];

  return (
    <Card>
      <div className="flex items-center justify-between">
        <Title>AI Monthly Insight</Title>
        <Button size="xs" loading={generate.isPending} onClick={() => generate.mutate()}>
          Generate for this month
        </Button>
      </div>
      {latest ? (
        <div className="mt-3">
          <p className="text-sm text-tremor-content">{latest.period}</p>
          <p className="mt-1">{latest.summary_text}</p>
        </div>
      ) : (
        <p className="mt-3 text-sm text-tremor-content">No insights yet — generate one.</p>
      )}
    </Card>
  );
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `npm test -- --run src/components` — Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add frontend/src/components
git commit -m "feat: add dashboard widgets (charts, anomalies, insights)"
```

---

### Task 4: Chat panel + full layout

**Files:**
- Create: `frontend/src/components/ChatPanel.tsx`
- Modify: `frontend/src/App.tsx` (compose the layout)
- Test: `frontend/src/components/ChatPanel.test.tsx`

**Interfaces:**
- Consumes: `sendChat`; all Task-3 widgets.
- Produces: `ChatPanel` — message list (user/assistant), input + send, per-answer tool chips ("aggregate_spend", …); final `App` grid layout.

- [ ] **Step 1: Write the failing test**

`frontend/src/components/ChatPanel.test.tsx`:
```tsx
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, expect, test, vi } from "vitest";
import ChatPanel from "./ChatPanel";

afterEach(() => vi.unstubAllGlobals());

test("sends a message and shows answer with tool chips", async () => {
  vi.stubGlobal("fetch", vi.fn().mockResolvedValue({
    ok: true,
    json: async () => ({
      answer: "You spent $412.30 on groceries in June.",
      tools_used: [{ tool: "aggregate_spend", args: { category: "Groceries" } }],
    }),
  }));

  render(<ChatPanel />);
  await userEvent.type(screen.getByPlaceholderText(/ask about your finances/i),
    "groceries in June?");
  await userEvent.click(screen.getByRole("button", { name: /send/i }));

  expect(await screen.findByText(/You spent \$412\.30/)).toBeInTheDocument();
  expect(screen.getByText("groceries in June?")).toBeInTheDocument();
  expect(screen.getByText("aggregate_spend")).toBeInTheDocument();
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `npm test -- --run src/components/ChatPanel` — Expected: FAIL (module doesn't exist).

- [ ] **Step 3: Implement**

`frontend/src/components/ChatPanel.tsx`:
```tsx
import { useState } from "react";
import { Badge, Button, Card, TextInput, Title } from "@tremor/react";
import { sendChat } from "../api/client";

interface Msg {
  role: "user" | "assistant";
  content: string;
  tools?: { tool: string; args: Record<string, unknown> }[];
}

export default function ChatPanel() {
  const [messages, setMessages] = useState<Msg[]>([]);
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);

  async function submit() {
    const question = input.trim();
    if (!question || busy) return;
    setInput("");
    setMessages((m) => [...m, { role: "user", content: question }]);
    setBusy(true);
    try {
      const reply = await sendChat(question);
      setMessages((m) => [...m, { role: "assistant", content: reply.answer, tools: reply.tools_used }]);
    } catch {
      setMessages((m) => [...m, { role: "assistant", content: "Sorry — something went wrong." }]);
    } finally {
      setBusy(false);
    }
  }

  return (
    <Card className="flex h-full flex-col">
      <Title>Ask your money anything</Title>
      <div className="mt-3 flex-1 space-y-3 overflow-y-auto">
        {messages.map((m, i) => (
          <div key={i} className={m.role === "user" ? "text-right" : ""}>
            <p className={`inline-block rounded px-3 py-2 text-sm ${
                m.role === "user" ? "bg-blue-900/50" : "bg-slate-800"}`}>
              {m.content}
            </p>
            {m.tools && m.tools.length > 0 && (
              <div className="mt-1 space-x-1">
                {m.tools.map((t, j) => <Badge key={j} color="emerald">{t.tool}</Badge>)}
              </div>
            )}
          </div>
        ))}
      </div>
      <form className="mt-3 flex gap-2"
            onSubmit={(e) => { e.preventDefault(); void submit(); }}>
        <TextInput placeholder="Ask about your finances…" value={input}
                   onValueChange={setInput} />
        <Button type="submit" loading={busy}>Send</Button>
      </form>
    </Card>
  );
}
```

Update `frontend/src/App.tsx` to compose everything:
```tsx
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import AnomalyList from "./components/AnomalyList";
import ChatPanel from "./components/ChatPanel";
import InsightCard from "./components/InsightCard";
import MonthlyTrend from "./components/MonthlyTrend";
import SpendByCategory from "./components/SpendByCategory";

const queryClient = new QueryClient();

export default function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <main className="min-h-screen bg-slate-950 p-6 text-slate-100">
        <h1 className="text-2xl font-semibold">AI Financial Insights</h1>
        <p className="mt-1 text-sm text-slate-400">
          RAG-grounded answers, anomaly alerts, and plain-English summaries of your money.
        </p>
        <div className="mt-6 grid gap-4 lg:grid-cols-3">
          <div className="space-y-4 lg:col-span-2">
            <div className="grid gap-4 md:grid-cols-2">
              <SpendByCategory />
              <MonthlyTrend />
            </div>
            <InsightCard />
            <AnomalyList />
          </div>
          <div className="lg:col-span-1"><ChatPanel /></div>
        </div>
      </main>
    </QueryClientProvider>
  );
}
```

- [ ] **Step 4: Run the full frontend suite**

Run: `npm test -- --run` — Expected: all pass (shell test still green with new layout).

- [ ] **Step 5: Manual smoke (backend running)**

With `uvicorn app.main:app` up and DB seeded: `npm run dev`, open the page — charts populate, chat answers with tool chips.

- [ ] **Step 6: Commit**

```bash
git add frontend/src
git commit -m "feat: add chat panel and full dashboard layout"
```

---

### Task 5: Project README + finishing touches

**Files:**
- Create: `README.md` (repo root)

**Interfaces:** none — documentation.

- [ ] **Step 1: Write the README**

`README.md` must contain, in order (write real content, not placeholders — pull the measured eval numbers from `docs/eval-results.md` if it exists, otherwise mark the eval section "run `python -m app.eval.run_eval` to produce"):

1. Title + one-paragraph pitch (hybrid agentic RAG over personal finance data).
2. Architecture diagram (ASCII, from the design spec) and a bullet on why hybrid RAG beats naive vector RAG for money questions.
3. **Measured eval results** — the A vs B table/percentages.
4. Quickstart:
   ```
   # backend
   cd backend && pip install -e ".[dev]"
   # db: local PostgreSQL with finuser/finpass/findb (see docs), then:
   alembic upgrade head && python -m app.seed
   uvicorn app.main:app --reload
   # frontend
   cd frontend && npm install && npm run dev
   ```
   Plus `.env` setup from `.env.example` (never commit real keys).
5. Tech stack table; project layout tree; testing (`pytest`, `npm test -- --run`); "Production-scale swaps" note (pgvector + Docker).

- [ ] **Step 2: Verify accuracy**

Every command in the README must have been actually run during Plans 1–4. Fix any drift found — the README is the first thing reviewers read.

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "docs: add project README with measured eval results"
```

---

## Self-Review

**Spec coverage (Plan 4 slice):** dashboard charts (category donut, monthly trend) ✓; anomaly alerts with explanations ✓; AI insight cards + generate action ✓; RAG chat UI surfacing tools used ✓; clean SPA, dark theme ✓; README/polish ✓ (Task 5). Deployment stretch remains optional/out of scope.

**Placeholder scan:** Task 5 defines README content requirements rather than verbatim text (content depends on measured eval results — a genuine runtime input, not a placeholder). All code steps carry complete code. ✓

**Type consistency:** `client.ts` types mirror Plan 2/3 response shapes exactly (`by_category`, `anomalies`, `insights`, `tools_used`); widget query keys unique; `ChatReply.tools_used` shape matches `ChatPanel` and the test. Version pins consistent (React 18 + Tremor 3 + Tailwind 3). ✓
