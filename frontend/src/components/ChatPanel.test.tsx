import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, expect, test, vi } from "vitest";
import ChatPanel from "./ChatPanel";

afterEach(() => vi.unstubAllGlobals());

test("sends a message and shows answer with tool chips", async () => {
  vi.stubGlobal(
    "fetch",
    vi.fn().mockResolvedValue({
      ok: true,
      json: async () => ({
        answer: "You spent $412.30 on groceries in June.",
        tools_used: [{ tool: "aggregate_spend", args: { category: "Groceries" } }],
      }),
    })
  );

  render(<ChatPanel />);
  await userEvent.type(
    screen.getByPlaceholderText(/ask about your finances/i),
    "groceries in June?"
  );
  await userEvent.click(screen.getByRole("button", { name: /send/i }));

  expect(await screen.findByText(/You spent \$412\.30/)).toBeInTheDocument();
  expect(screen.getByText("groceries in June?")).toBeInTheDocument();
  expect(screen.getByText("aggregate_spend")).toBeInTheDocument();
});
