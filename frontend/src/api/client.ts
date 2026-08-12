export interface CategoryTotal {
  category: string;
  total: number;
}
export interface MonthlyPoint {
  period: string;
  spend: number;
  income: number;
}
export interface Txn {
  id: number;
  date: string;
  merchant: string;
  amount: number;
  category: string;
  description: string;
  is_anomaly: boolean;
}
export interface Anomaly extends Txn {
  z_score: number;
  category_mean: number;
  explanation?: string;
}
export interface Insight {
  id: number;
  period: string;
  summary_text: string;
  generated_at: string | null;
}
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
  get<{ by_category: CategoryTotal[] }>("/stats/by-category").then(
    (d) => d.by_category
  );

export const getMonthly = () =>
  get<{ monthly: MonthlyPoint[] }>("/stats/monthly").then((d) => d.monthly);

export const getAnomalies = (explain = false) =>
  get<{ anomalies: Anomaly[] }>(`/anomalies?explain=${explain}`).then(
    (d) => d.anomalies
  );

export const getInsights = () =>
  get<{ insights: Insight[] }>("/insights").then((d) => d.insights);

export const generateInsight = (period: string) =>
  post<Insight>(`/insights/${period}/generate`, {});

export const getTransactions = (limit = 50) =>
  get<{ transactions: Txn[] }>(`/transactions?limit=${limit}`).then(
    (d) => d.transactions
  );

export const sendChat = (message: string, sessionId = "default") =>
  post<ChatReply>("/chat", { message, session_id: sessionId });
