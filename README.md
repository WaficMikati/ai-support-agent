# AI support agent

Reads a Chatwoot inbox, classifies each incoming message, and either answers
it from `knowledge.md` or issues a Stripe refund.

Built for a two-part workshop:

- **Session 1** — widget, polling, classification, answering from the
  knowledge file. Ends with a working responder that cannot act on anything.
- **Session 2** — the refund path, the approval rules, scoped credentials.

Demo only. The agent refuses to start against a live Stripe key.

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
      │               ├── passes → Stripe refund + reply + resolve
      │               └── fails  → private note, left open for a human
      │
      └── support → answer from knowledge.md → reply + resolve
```

Only the newest message decides whether we act: if we spoke last, the
conversation is skipped. That is what keeps a polling loop from replying
twice. A private note counts as us speaking, which is what stops a held
refund from being flagged again on every poll. Chatwoot's activity entries
(labels, assignments, status changes) are ignored, since otherwise a label
added after a customer writes in would look like a reply.

Answered and refunded conversations are resolved. Chatwoot reopens one as
soon as the customer writes again, so nothing is lost.

`knowledge.md` is re-read on every poll. Editing it changes the answers
within one interval, with no restart.

## Refund policy

Auto-approval requires **all** of:

| Condition | Default |
|---|---|
| Amount at or under | `MAX_AUTO_REFUND_CENTS` = $50.00 |
| Purchase within | `MAX_CHARGE_AGE_DAYS` = 30 days |
| Previous refunds on the account | none |
| Classifier confidence at or above | `MIN_CONFIDENCE` = 0.8 |
| Charge not already refunded, in full or in part | — |
| Exactly one refundable charge, so there is nothing to guess | — |

Anything else posts a private note with the reason and a suggested reply,
leaves the conversation open, and touches no money.

Two details that are easy to get wrong. Stripe sets `refunded` only when a
charge is refunded *in full*, so a partly refunded charge looks untouched;
both count as refunded here. And customers rarely say which payment they
mean, so if more than one charge could still be refunded the agent refuses to
choose.

## Setup

### 1. Chatwoot

```bash
cd deploy
cp env.example .env          # then set SECRET_KEY_BASE, POSTGRES_PASSWORD, REDIS_PASSWORD
docker compose run --rm rails bundle exec rails db:chatwoot_prepare
docker compose up -d
```

`deploy/docker-compose.yaml` is the upstream production compose with two
changes: Postgres reads its password from `.env`, and Redis no longer
publishes a host port, which would clash with a Redis already running
locally.

You need **two inboxes**:

- a **Web widget** inbox for the website chat
- an **API** inbox for anything that injects messages programmatically

That second one is not optional for testing: Chatwoot rejects incoming
messages on a widget inbox with *"Incoming messages are only allowed in Api
inboxes"*. The `/public/api/v1/inboxes/...` endpoints resolve the API
channel, not the widget, which has its own `/api/v1/widget/...` flow.

### 2. Email (optional)

The compose file includes a **GreenMail** container: a real IMAP and SMTP
server, so the email channel can be set up and proven without anybody's
mailbox credentials. Mailboxes are `support@chatwoot.test` (login `support`)
and `customer@chatwoot.test` (login `customer`), on ports 3025 for SMTP and
3143 for IMAP.

Create an Email inbox and point IMAP and SMTP at host `greenmail`. Three
things are easy to get wrong here, all of them found the hard way:

- **Set `imap_authentication` to `login`, not the default `plain`.** Chatwoot's
  default issues `AUTHENTICATE PLAIN`, which many servers, GreenMail included,
  do not advertise. `login` uses the plain `LOGIN` command instead. Chatwoot
  accepts `plain`, `login` or `cram-md5` per inbox.
- **The IMAP login is not always the email address.** For GreenMail it is
  `support`, and logging in with the full address makes it try to create a
  second user and drop the connection.
- **Test emails need a `Message-ID` header.** Chatwoot silently discards mail
  without one, with no error anywhere. Real clients always set it; hand-built
  test messages often do not.

Chatwoot fetches on a schedule, so email is not instant. To pull immediately:

```bash
docker compose exec rails bundle exec rails runner \
  'Inboxes::FetchImapEmailsJob.perform_now(Channel::Email.find_by(email: "support@chatwoot.test"))'
```

Note that Chatwoot passes the message through **unmodified**: signatures and
quoted reply chains reach the classifier intact. That tested fine, but it is
worth knowing that the model sees the whole thing.

Swapping in a real mailbox is a change of host, port, login and password on
the inbox. Nothing in the agent changes.

### 3. The agent

```bash
uv sync
cp .env.example .env         # fill it in; agent.py reads it, nothing to export
uv run python agent.py
```

## Tests

```bash
uv run pytest                          # 68 tests, no network, no credentials
uv run python scripts/live_check.py    # the inbox adapter against live Chatwoot
uv run python scripts/e2e_check.py     # everything live: Groq, Stripe test mode, Chatwoot
uv run python scripts/reload_check.py  # editing knowledge.md with the agent running
```

`pytest` fakes Chatwoot, Stripe and the model, so policy, routing, the three
adapter mappings and the loop are checked in isolation.

The scripts need a running Chatwoot and a filled-in `.env`. `e2e_check.py`
additionally needs `STRIPE_SETUP_KEY` to create the customer and charge that a
refund test requires, because the agent's own key deliberately cannot.
`reload_check.py` starts the agent, edits `knowledge.md` underneath it, and
restores the file afterwards.

## Credentials

- **Groq** — console.groq.com, free tier, no card. Model
  `openai/gpt-oss-20b`, one of the two Groq models supporting strict JSON
  schema output. Rate limits are per organisation, not per key.
- **Stripe** — a restricted key (`rk_test_`) created in a sandbox, with
  **Charges and refunds: write** and **Customers: read**, everything else
  None. Stripe has no standalone refunds permission, so write on "Charges
  and refunds" is the narrowest grant that allows issuing one.
- **Chatwoot** — Profile Settings → Access Token.
