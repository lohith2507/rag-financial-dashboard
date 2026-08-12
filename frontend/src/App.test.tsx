import { render, screen } from "@testing-library/react";
import App from "./App";

test("renders the app heading", () => {
  render(<App />);
  expect(
    screen.getByRole("heading", { name: /AI Financial Insights/i })
  ).toBeInTheDocument();
});
