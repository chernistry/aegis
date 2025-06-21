import Chat, { Bubble, useMessages } from '@chatui/core';
import '@chatui/core/es/styles/index.css';

const API_URL = import.meta.env.VITE_API || 'http://localhost:8910';

function App() {
  const { messages, appendMsg, setTyping } = useMessages([]);

  async function handleSend(type: string, val: string) {
    if (type !== 'text' || !val.trim()) return;

    appendMsg({ type: 'text', content: { text: val }, position: 'right' });
    setTyping(true);

    let answer = '';
    try {
      const resp = await fetch(`${API_URL}/chat/stream`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question: val, top_k: 5 }),
      });
      if (!resp.body) throw new Error('No response body');
      const reader = resp.body.getReader();
      const decoder = new TextDecoder();
      let buf = '';
      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        buf += decoder.decode(value, { stream: true });
        const parts = buf.split('\n\n');
        buf = parts.pop() || '';
        for (const part of parts) {
          if (part.startsWith('data: ')) {
            answer += part.slice(6);
          }
        }
      }
    } catch (err) {
      answer = `⚠️ Ошибка: ${(err as Error).message}`;
    }

    appendMsg({ type: 'text', content: { text: answer }, position: 'left' });
    setTyping(false);
  }

  return (
    <Chat
      navbar={{ title: 'Aegis' }}
      messages={messages}
      onSend={handleSend}
      renderMessageContent={(m) => <Bubble content={m.content.text} />}
    />
  );
}

export default App;
