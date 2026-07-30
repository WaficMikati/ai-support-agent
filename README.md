# AI support agent

Reads a Chatwoot inbox, classifies each incoming message, and either answers
it from a help centre or issues a Stripe refund.

Built for a two-part workshop:

- **Session 1** — widget, polling, classification, answering from documentation.
  Ends with a working responder that cannot act on anything.
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
the conversation so far + knowledge.md + matching articles
      │
      ▼
understand → {reply, refund_requested, confidence}    one model call
      │
      ├── refund_requested → refund_decision()   pure function, no model
      │        ├── passes → Stripe refund, then a reply stating the amount
      │        └── fails  → the model's reply, plus a private note, left open
      │
      └── otherwise → the model's reply + resolve
```

**The model proposes; code decides.** It reads the thread and writes the next
reply, and says separately whether a refund is being requested. Whether money
moves is settled afterwards against thresholds it never sees.

This replaced a design that reduced each message to one of two labels before
anything was understood, and it was replaced for two reasons. It could not hold a
conversation, because only the newest message was ever sent: "it was due last
Tuesday" is unanswerable on its own. And every distinction had to be written into
the rubric in advance, which does not converge. "How do I get a refund?" was read
as a request and refunded twenty dollars, and adding a counter-example only
postpones the next phrasing. Now that question is simply answered from the
refunds article, because nothing forces a branch before the model has read
anything.

Only the newest message decides whether we act: if we spoke last, the
conversation is skipped. That is what keeps a polling loop from replying
twice. A private note counts as us speaking, which is what stops a held
refund from being flagged again on every poll. Chatwoot's activity entries
(labels, assignments, status changes) are ignored, since otherwise a label
added after a customer writes in would look like a reply.

That rule is correct but not instantaneous, so two more things guard against
acting on the same message twice:

- **The last handled message id is recorded on the Chatwoot conversation**, in
  its `custom_attributes`, so the memory belongs to Chatwoot rather than to this
  process. That is what survives a restart and is shared with any other copy of
  the agent. An in-process set is kept as well, purely as the cheap first check
  within a single run. Both are written only after the work succeeds, so a
  transient failure is retried rather than swallowed.

  This matters more than it sounds. With the set alone, a running agent and a
  check script both answered the same conversations and posted duplicate
  replies, because each process had its own memory. Verified with the marker: a
  fresh process handles the message and records id 444; a second fresh process
  then makes zero model calls and posts nothing.

- **Refunds carry a Stripe idempotency key** built from the conversation and the
  charge, `refund-conv482-ch_3Tym…`. Keyed on the charge rather than the message
  that asked, because somebody asking twice in one conversation gives two
  message ids, and two keys mean Stripe treats the second attempt as a new
  refund. Keyed on the charge, the second attempt returns the first refund. The
  policy would usually catch it first, since the charge then reads as refunded,
  but that is a check racing a write rather than a guarantee. Stripe keeps keys
  for at least 24 hours.

Answered and refunded conversations are resolved. Chatwoot reopens one as
soon as the customer writes again, so nothing is lost.

## Where answers come from

Two separate inputs, which is what lets the documentation change without
touching the instructions:

- **`knowledge.md` is how to behave.** Tone, when to ask a follow-up rather
  than hand over, and the standing rule never to invent. It states no facts at
  all: no prices, no policies, not even where a setting lives. It used to, and
  that was a bug rather than duplication, because it said cancellation was under
  Billing while the documentation said Subscription, leaving the agent two
  contradictory answers to pick from. It is re-read on every poll, so editing it
  changes behaviour within one interval, no restart.
- **The help centre is what may be stated as fact.** Articles are fetched from
  a Chatwoot portal, cached for five minutes, and the few relevant to the
  question go into the prompt. If nothing matches, the agent is told to answer
  generally and to state nothing specific about the product.

That split is the point. The model already knows how to write a support reply;
what it cannot know is your prices and policies. So it brings the language and
the help centre brings the facts. Ask it something general and it answers.
Ask it something about the product with no matching article and it declines
rather than inventing.

Selection is word overlap against the fetched articles, with a title match
weighted above a body match, and words matching on a shared prefix so "cancel"
finds "Cancelling your subscription". Chatwoot's own `?query=` is a phrase
match and is not usable for this: it returns nothing for "How do I cancel my
subscription?" despite that article existing. Fetching a few dozen short
articles and scoring them locally is the right amount of machinery at this
size; a corpus too large to fetch is where embeddings start to earn their keep.

`scripts/seed_helpcenter.py` creates a mock help centre for a fictional coffee
subscription, so the workshop has documentation to answer from that cannot be
mistaken for anyone's real policy. Point `HELP_CENTRE_PORTAL` at a different
portal to use real documentation instead; no code changes.

**The poll costs one request, not one per conversation.** The conversation list
already carries the newest message, which is all the loop needs to know who
spoke last, so the messages of each conversation are fetched only when that
newest entry is one of Chatwoot's own activity or template records and the loop
has to look further back.

This matters because refunds held for a human stay open by design, so that queue
only grows. Fetching per conversation made the agent slower the longer people
were behind: measured here, eight held refunds took a reply from about 2 seconds
to 3.6 to 4.6 seconds. With the list-first approach, ten held refunds in the
queue still reply in 1.4 to 2.9 seconds, of which about one second is the model.
Across 43 polls with ten open conversations it made four message fetches; the
per-conversation version would have made about 430.

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
| Rubric score at or above | `MIN_CONFIDENCE` = 0.6, two of three points |
| Charge not already refunded, in full or in part | — |
| Exactly one refundable charge, so there is nothing to guess | — |

Anything else tells the customer a colleague will follow up, posts a private
note with the reason, leaves the conversation open, and touches no money. The
customer-facing acknowledgement matters: posting only the note leaves them
staring at silence, which is indistinguishable from a broken agent, and it is
the single most confusing thing to hit while demonstrating this. The
acknowledgement deliberately says nothing about the outcome, because whether to
refund is the human's decision.

**Confidence is scored, not self-reported.** The model is never asked how sure
it is. It answers three plain questions about the message instead, and the score
is arithmetic in code:

| Signal | Question |
|---|---|
| `clear_request` | did they plainly ask, rather than wonder aloud? |
| `charge_identified` | did they say which payment, by date, amount or description? |
| `hedging` | did they hedge, with "maybe", "I think", "would I be able to"? |

A model asked to rate its own confidence is introspecting, which it is bad at,
and it returned a number that looked authoritative while meaning little. These
are observable. They also make the note to a colleague specific: "hedging: yes"
rather than "confidence 0.67".

The threshold is two of the three points. That arithmetic decides real
behaviour: a plain "I want my money back" scores clear request and no hedging
but names no charge, landing on 0.67 and approved, while "maybe I could get a
refund?" loses the hedging point too and goes to a human. Requiring all three
would hold nearly every genuine request, because customers hardly ever name the
payment.

Two more details that are easy to get wrong. Stripe sets `refunded` only when a
charge is refunded *in full*, so a partly refunded charge looks untouched; both
count as refunded here. And customers rarely say which payment they mean, so if
more than one charge could still be refunded the agent refuses to choose.

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

**Turn off auto-offline, or the widget tells visitors nobody is there.**
Chatwoot decides online or away from human agent presence, meaning a live
dashboard websocket. This agent authenticates with an API token and never opens
one, so by default the widget greets visitors with *"We are away at the moment.
We will be back as soon as possible"* — from something that answers in seconds.

```bash
docker compose exec rails bundle exec rails runner \
  'AccountUser.find_by(user_id: User.find_by(email: "ops@example.com").id).update!(auto_offline: false, availability: :online)'
```

The widget then shows a green dot and "We are online". The reply-time line
beneath it is a separate per-inbox setting whose fastest option is "in a few
minutes", so it understates the agent and cannot be improved without patching
the widget.

**Turn off rate limiting, or repeated resets stop the widget loading at all.**
Chatwoot throttles its widget endpoints per IP, and everything in a local demo
comes from one address. Reloading and resetting a handful of times in quick
succession earns a 429 on `/widget`, after which the chat bubble simply does not
appear, with nothing in the UI to say why. Add `ENABLE_RACK_ATTACK=false` to
`deploy/.env` and restart `rails`.

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
uv run python scripts/seed_helpcenter.py   # mock documentation to answer from
uv run python agent.py
```

### 4. The demo page

```bash
uv run python demo/serve.py    # then open http://localhost:8080
```

A one-file page carrying the widget snippet, which is also what a workshop
attendee ends up with. It identifies the visitor the way a real site does for a
signed-in customer, because an anonymous visitor has no email and the refund
path matches customers by email.

Its **Start a fresh conversation** button needs the server, which is why this is
not `python -m http.server`. Resetting the chat is harder than it looks, and
four things do not work:

- `$chatwoot.reset()` leaves the visitor token byte for byte identical
- deleting the cookie from the page does nothing, because the chat is an iframe
  on its own origin and keeps its own copy
- rotating the identifier with `setUser` does give a new conversation, but the
  new contact cannot reuse the old email, since Chatwoot allows one contact per
  email per account, and without an email refunds stop matching
- deleting the contacts frees the email, but then the page holds a token
  pointing at a deleted contact, `setUser` fails silently, and the visitor ends
  up anonymous

So the button asks the server to resolve every open conversation, mint a whole
new customer with its own refundable charge, identify as them, and then
**reload**. Clearing the queue matters for more than tidiness: held refunds stay
open by design and every open conversation is one the agent reconsiders on each
poll. They are resolved rather than deleted, so the history stays visible in the
dashboard. The reload is
required rather than cosmetic: `setUser` swaps the identity but the widget's
websocket stays subscribed to the previous one, so messages still send while
replies are published to a channel the page is no longer listening on. The
visitor sees silence while the agent believes it answered. The Stripe key stays
in the server process rather than being handed to the browser.

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
