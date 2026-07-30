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
classify → {intent, confidence}          OpenRouter, strict JSON schema
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

That rule is correct but not instantaneous, so two more things guard against
acting on the same message twice:

- **The agent remembers the message ids it has handled.** Between classifying
  a message and posting the reply, the customer's message is still the newest
  thing in the conversation, so an overlapping pass, or a second copy of the
  agent, would pick it up again. Ids are recorded only after the work
  succeeds, so a transient failure is retried rather than swallowed. This is
  in memory, and therefore gone on restart.
- **Refunds carry a Stripe idempotency key** derived from the conversation and
  message id, `refund-conv482-msg91`. If the same request ever reaches Stripe
  twice, whether from a race or from a crash between issuing the refund and
  posting the reply, Stripe returns the original refund instead of issuing a
  second one. This is the backstop that survives a restart, and it is what
  actually protects the money. Stripe keeps keys for at least 24 hours.

Answered and refunded conversations are resolved. Chatwoot reopens one as
soon as the customer writes again, so nothing is lost.

`knowledge.md` is re-read on every poll. Editing it changes the answers
within one interval, with no restart.

**Temperature is pinned to 0.** Both jobs want the most likely answer rather
than a creative one, and this is not cosmetic: left at the provider default,
a 20B model occasionally dropped a stray Chinese or Arabic character into an
otherwise correct English support reply, and the classifier returned 0.90,
0.95 and 0.98 on byte-identical input. At 0 the classification repeats exactly
and the replies come out clean. `MODEL_TEMPERATURE` overrides it.

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
uv run pytest                          # 102 tests, no network, no credentials
uv run python scripts/live_check.py    # the inbox adapter against live Chatwoot
uv run python scripts/e2e_check.py     # everything live: OpenRouter, Stripe, Chatwoot
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

- **OpenRouter** — openrouter.ai, free, no card. Model
  `openai/gpt-oss-20b:free`. Current free models supporting structured output:
  `https://openrouter.ai/api/v1/models?supported_parameters=structured_outputs`

  **Do not rely on the schema alone.** OpenRouter serves a model through
  several provider endpoints and only some enforce `strict: true` with
  constrained decoding; the rest generate JSON by following instructions.
  Requests send `provider: {require_parameters: true}`, but that filters on
  what a provider *advertises*, not on what it does. So the classifier prompt
  also spells out the JSON contract, and an unusable reply is retried. An
  earlier prompt said "reply with refund" and never mentioned JSON, relying
  entirely on enforcement: a non-enforcing endpoint followed the words and
  returned prose, which the enforcing one had been quietly covering up. Belt
  and braces is also what makes the model and provider swappable.

  `OPENROUTER_PROVIDER=Groq,Fireworks` pins the endpoint and disables
  fallbacks, if you ever need to guarantee which one serves a request.

  **Mind the quota.** Per OpenRouter's own rate-limit docs, `:free` models
  allow 20 requests a minute and **50 a day in total** on an account that has
  never bought credit, rising to 1,000 a day once $10 has been purchased at any
  point. The agent spends one request classifying and a second writing the
  answer, so roughly 25 conversations a day at the free ceiling.

  That daily cap applies only to `:free` models. Paid models have no daily
  request limit, only a credit balance, and are cheap here: a conversation
  measures around 600 tokens across both calls. So $10 both lifts the free cap
  and covers tens of thousands of conversations on the paid variant.

  `OPENROUTER_BASE_URL` and `OPENROUTER_MODEL` accept any OpenAI-shaped
  endpoint, so Groq, whose free tier allows thousands of requests a day,
  remains a two-line change.
- **Stripe** — a restricted key (`rk_test_`) created in a sandbox, with
  **Charges and refunds: write** and **Customers: read**, everything else
  None. Stripe has no standalone refunds permission, so write on "Charges
  and refunds" is the narrowest grant that allows issuing one.
- **Chatwoot** — Profile Settings → Access Token.
