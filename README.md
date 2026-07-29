# AI support agent

Reads a Chatwoot inbox, classifies each incoming message, and either answers
it from `knowledge.md` or issues a Stripe refund.

Built for a two-part workshop:

- **Session 1** — widget, polling, classification, answering from the
  knowledge file. Ends with a working responder that cannot act on anything.
- **Session 2** — the refund path, the approval rules, scoped credentials.

## Design

The model decides *what the customer is asking for*. Plain code decides
*whether we give them money*. Those two are never mixed.

```
Chatwoot widget
      │
      ▼
GET /conversations?status=open          poll, no webhooks, no tunnel
      │
      ▼
classify → {intent, confidence}          Groq, strict JSON schema
      │
      ├── refund  → refund_decision()    pure function, no model
      │               ├── passes → Stripe refund + reply
      │               └── fails  → private note for a human
      │
      └── support → answer from knowledge.md → reply
```

Only the newest message decides whether we act: if we spoke last, the
conversation is skipped. That is what keeps a polling loop from replying
twice.

## Refund policy

Auto-approval requires **all** of:

| Condition | Default |
|---|---|
| Amount at or under | `MAX_AUTO_REFUND_CENTS` = $50.00 |
| Purchase within | `MAX_CHARGE_AGE_DAYS` = 30 days |
| Previous refunds | none |
| Classifier confidence at or above | `MIN_CONFIDENCE` = 0.8 |
| Charge not already refunded | — |

Anything else posts a private note in the inbox with the reason and a
suggested reply, and waits for a person.

## Running it

```bash
uv sync
cp .env.example .env    # fill in the four values
uv run python agent.py
```

## Tests

```bash
uv run pytest
```

No network, no credentials. Chatwoot, Stripe and the model are all faked, so
the refund policy and routing are checked in isolation.

## Credentials

- **Groq** — console.groq.com, free tier, no card. Model `openai/gpt-oss-20b`,
  which is one of the two Groq models supporting strict JSON schema output.
- **Stripe** — a restricted key (`rk_test_`), refunds write plus charges and
  customers read. Nothing else. Create it in a sandbox first.
- **Chatwoot** — Profile Settings → Access Token.
