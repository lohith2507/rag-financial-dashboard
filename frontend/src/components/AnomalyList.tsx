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
      {data.length === 0 && (
        <p className="mt-3 text-sm text-tremor-content">No anomalies detected.</p>
      )}
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
