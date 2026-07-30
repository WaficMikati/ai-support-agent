**Chatwoot integration**

- Self-hosted via Docker Compose: Rails, Sidekiq, Postgres, Redis. Running on my machine on port 3000, not exposed publicly. Chatwoot itself is unmodified: no plugin, no fork, and its own AI features are off.
- Two inboxes. A **web widget** inbox for the chat on the page, and an **API** inbox for pushing messages in programmatically. Both are needed: Chatwoot rejects incoming messages on a widget inbox.
- The agent is a separate process, not something inside Chatwoot. It authenticates with one header, `api_access_token`, taken from Profile Settings, and uses four endpoints:
  - `GET /conversations?status=open` — poll for anything waiting
  - `GET /conversations/{id}/messages` — read the thread
  - `POST /conversations/{id}/messages` — reply, or leave an internal private note
  - `POST /conversations/{id}/toggle_status` — resolve once handled
- **Polling, not webhooks.** Nothing needs a public URL, a tunnel, or an open port, which is what keeps it reproducible on any laptop.
- Division of responsibility: Chatwoot holds the conversations, the markdown file holds the guidance, Stripe performs the action. The agent is the only piece that knows about all three, so any one of them can be swapped without touching the others.
- Escalation to a human is a Chatwoot private note. Staff see the reason it was held plus a suggested reply; the customer sees nothing, and the conversation stays open.

**It is not connected to 4geeks.com.** It is standalone and local, and it only talks to Stripe in test mode.
