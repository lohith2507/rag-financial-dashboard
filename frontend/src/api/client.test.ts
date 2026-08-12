import { afterEach, expect, test, vi } from "vitest";
import { getAnomalies, getByCategory, sendChat } from "./client";

function stubFetch(body: unknown, ok = true) {
  const fn = vi.fn().mockResolvedValue({
    ok,
    status: ok ? 200 : 500,
    json: async () => body,
  });
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
    message: "how much?",
    session_id: "s1",
  });
});

test("errors throw with status", async () => {
  stubFetch({}, false);
  await expect(getByCategory()).rejects.toThrow(/500/);
});
