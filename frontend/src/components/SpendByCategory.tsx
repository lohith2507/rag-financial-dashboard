import { useQuery } from "@tanstack/react-query";
import { Card, DonutChart, Title } from "@tremor/react";
import { getByCategory } from "../api/client";

export default function SpendByCategory() {
  const { data = [] } = useQuery({
    queryKey: ["by-category"],
    queryFn: () => getByCategory(),
  });
  return (
    <Card>
      <Title>Spend by Category</Title>
      <DonutChart
        className="mt-4 h-52"
        data={data}
        category="total"
        index="category"
        valueFormatter={(v) => `$${v.toLocaleString()}`}
      />
    </Card>
  );
}
