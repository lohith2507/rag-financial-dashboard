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
      setMessages((m) => [
        ...m,
        { role: "assistant", content: reply.answer, tools: reply.tools_used },
      ]);
    } catch {
      setMessages((m) => [
        ...m,
        { role: "assistant", content: "Sorry — something went wrong." },
      ]);
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
            <p
              className={`inline-block rounded px-3 py-2 text-sm ${
                m.role === "user" ? "bg-blue-900/50" : "bg-slate-800"
              }`}
            >
              {m.content}
            </p>
            {m.tools && m.tools.length > 0 && (
              <div className="mt-1 space-x-1">
                {m.tools.map((t, j) => (
                  <Badge key={j} color="emerald">
                    {t.tool}
                  </Badge>
                ))}
              </div>
            )}
          </div>
        ))}
      </div>
      <form
        className="mt-3 flex gap-2"
        onSubmit={(e) => {
          e.preventDefault();
          void submit();
        }}
      >
        <TextInput
          placeholder="Ask about your finances…"
          value={input}
          onValueChange={setInput}
        />
        <Button type="submit" loading={busy}>
          Send
        </Button>
      </form>
    </Card>
  );
}
