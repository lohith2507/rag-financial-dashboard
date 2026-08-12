import { useQuery } from "@tanstack/react-query";
import { BarChart, Card, Title } from "@tremor/react";
import { getMonthly } from "../api/client";

export default function MonthlyTrend() {
  const { data = [] } = useQuery({ queryKey: ["monthly"], queryFn: getMonthly });
  return (
    <Card>
      <Title>Spend vs Income by Month</Title>
      <BarChart
        className="mt-4 h-52"
        data={data}
        index="period"
        categories={["spend", "income"]}
        valueFormatter={(v) => `$${v.toLocaleString()}`}
      />
    </Card>
  );
}
