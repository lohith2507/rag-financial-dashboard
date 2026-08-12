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
          RAG-grounded answers, anomaly alerts, and plain-English summaries of your
          money.
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
          <div className="lg:col-span-1">
            <ChatPanel />
          </div>
        </div>
      </main>
    </QueryClientProvider>
  );
}
