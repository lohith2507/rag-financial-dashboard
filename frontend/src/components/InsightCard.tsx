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
        <p className="mt-3 text-sm text-tremor-content">
          No insights yet — generate one.
        </p>
      )}
    </Card>
  );
}
