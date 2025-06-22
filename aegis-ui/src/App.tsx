import Chat, { Bubble, useMessages } from '@chatui/core';
import type { MessageProps } from '@chatui/core';
import '@chatui/core/dist/index.css';
import './App.css';
import { useState } from 'react';
import ReactMarkdown from 'react-markdown';

const API_URL = import.meta.env.VITE_API || 'http://localhost:8910';

// Определяем тип MessageWithoutId для корректной типизации
type MessageWithoutId = Omit<MessageProps, '_id'>;

// Предварительное сообщение от ассистента
const initialMessages: MessageWithoutId[] = [
  {
    type: 'text',
    content: {
      text: 'Привет! Я Aegis, ваш ИИ-ассистент. Чем могу помочь?'
    },
    position: 'left'
  }
];

function App() {
  const { messages, appendMsg } = useMessages(initialMessages);
  const [isTyping, setIsTyping] = useState(false);

  async function handleSend(type: string, val: string) {
    if (type !== 'text' || !val.trim()) return;

    appendMsg({ type: 'text', content: { text: val }, position: 'right' });
    setIsTyping(true);

    try {
      const resp = await fetch(`${API_URL}/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question: val }),
      });

      if (!resp.ok) throw new Error(`HTTP ошибка: ${resp.status}`);
      
      const chatResponse = await resp.json();
      
      let answer = chatResponse.answer;
      if (chatResponse.sources && chatResponse.sources.length > 0) {
        const sourcesText = chatResponse.sources.map((s: any) => `- ${s.source} (Score: ${s.score.toFixed(2)})`).join('\n');
        answer += `\n\n**Источники:**\n${sourcesText}`;
      }
      
      appendMsg({ type: 'text', content: { text: answer }, position: 'left' });

    } catch (err) {
      const errorMsg = `⚠️ Ошибка: ${(err as Error).message}`;
      appendMsg({ type: 'text', content: { text: errorMsg }, position: 'left' });
    } finally {
      setIsTyping(false);
    }
  }

  function renderMessageContent(msg: MessageProps) {
    const { content } = msg;
    // @ts-ignore
    return <Bubble content={<ReactMarkdown>{content.text}</ReactMarkdown>} />;
  }

  return (
    <div className="app-container">
      <Chat
        navbar={{ title: 'Aegis' }}
        messages={messages}
        renderMessageContent={renderMessageContent}
        onSend={handleSend}
        locale="ru-RU"
        placeholder="Введите сообщение..."
        isTyping={isTyping}
      />
    </div>
  );
}

export default App;
