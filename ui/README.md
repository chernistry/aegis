# Aegis RAG - React Chat UI

This directory will contain the React-based chat interface for the Aegis RAG project.

## Tech Stack
- **Framework:** React (Next.js or Vite)
- **UI Kit:** [HuggingFace Chat UI](https://github.com/huggingface/chat-ui), [Alibaba ChatUI](https://chatui.io/), or a custom build with [shadcn/ui](https://ui.shadcn.com/).
- **State Management:** Zustand or React Query
- **Styling:** Tailwind CSS

## Getting Started (Next Steps)

1.  **Initialize Project:** Use `npx create-next-app@latest .` or `npx create-vite@latest . --template react-ts` to scaffold the project here.
2.  **Install Dependencies:** Add the chosen UI kit, Tailwind CSS, and other necessary libraries.
3.  **Implement SSE Client:** Create a component that connects to the `http://localhost:8910/chat/stream` endpoint and renders the streaming response.
4.  **Add to Docker:** Create a `Dockerfile` for the UI and add the service to the main `docker-compose.yml`. 