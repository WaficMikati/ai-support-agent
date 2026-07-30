# AI support agent

A customer support agent that reads a shared inbox, answers questions from your
documentation, and issues refunds — with the decision about whether money moves
kept firmly out of the model's hands.

It is small on purpose. One Python file, no agent framework, and every external
system reached over plain HTTP so you can see what is being sent.

Built as the worked example for a two-part workshop:

- **Session one** — the widget, the polling loop, and answering from
  documentation. Ends with a working agent on your own page that cannot act on
  anything.
- **Session two** — the refund path: the rules, the human hand-off, and scoped
  credentials that only permit one operation.

It is a demo. It refuses to start against a live Stripe key.

---

## What it actually does

A visitor types into a chat widget on a web page. That lands in
[Chatwoot](https://www.chatwoot.com/), an open source shared inbox. The agent
polls Chatwoot, reads the conversation, and replies — or refunds them.

```
Chatwoot  ──poll──▶  the conversation so far
                     + how to behave (knowledge.md)
                     + the articles that match (your help centre)
                            │
                            ▼
                     one model call
                            │
                     {reply, refund_requested, three rubric signals}
                            │
              ┌─────────────┴─────────────┐
     refund requested                 everything else
              │                             │
      apply the policy                 send the reply
      (code, no model)                  and resolve
              │
    ┌─────────┴─────────┐
  passes              fails
    │                   │
 refund via         reply, plus a private
 Stripe, then       note for a colleague,
 say the amount     conversation left open
```

**The model proposes; code decides.** It reads the thread and writes the next
reply, and separately says whether the customer is asking for a refund. Whether
that refund actually happens is settled afterwards, in code, against thresholds
the model never sees.

That flow is written as a graph, because the routing is the interesting part:

```python
NODES = {"understand": ..., "answer": ..., "refund": ..., "execute_refund": ..., "hold": ...}

EDGES = {
    "understand": lambda s: "refund" if s.proposal.refund_requested else "answer",
    "refund":     lambda s: "execute_refund" if s.decision.auto_approve else "hold",
    "answer": lambda s: None, "execute_refund": lambda s: None, "hold": lambda s: None,
}

def run_graph(state, start="understand"):
    node = start
    while node:
        state = NODES[node](state)
        node = EDGES[node](state)
    return state
```

No framework, and none needed: a graph is a dict of functions and a dict of
routing rules. The edge out of `understand` is the model's choice. The edge out
of `refund` belongs to `refund_decision`. The model can ask for money to move; it
cannot make it move.

---

## Running it

### What you need first

| | |
|---|---|
| **Docker** with Compose v2 | Chatwoot runs in containers. Around 4 GB of RAM. |
| **Python 3.11+** and [uv](https://docs.astral.sh/uv/) | `curl -LsSf https://astral.sh/uv/install.sh \| sh` |
| **A model API key** | [openrouter.ai](https://openrouter.ai) is free with no card. Anything OpenAI-shaped works, including Groq. |
| **A Stripe sandbox** | Free. You need a restricted key, described below. |

Everything else is created for you.

It binds five ports on localhost: `3000` Chatwoot, `8080` the demo page, `5432`
Postgres, and `3025` / `3143` for mail. Free them first if something else has
them, since Docker's failure here is not always obvious.

### 0. Get it

```bash
git clone git@github.com:WaficMikati/ai-support-agent.git
cd ai-support-agent
```

### 1. Start Chatwoot

```bash
cd deploy
cp env.example .env
```

`deploy/env.example` is Chatwoot's own, several hundred commented lines of it.
Three keys in it are blank or placeholder and have to be filled in. The secrets
are yours to invent; nothing outside your machine sees them.

```
SECRET_KEY_BASE=<long random string: openssl rand -hex 64>
POSTGRES_PASSWORD=<any password>
REDIS_PASSWORD=<any password>
```

Keep `SECRET_KEY_BASE` alphanumeric, as the file itself warns. Hex satisfies
that; a password manager's punctuation may not.

Then add one line that is **not** in the file, since Chatwoot ships it commented
out and defaulting to on:

```
ENABLE_RACK_ATTACK=false
```

This matters for a demo. Chatwoot rate-limits its widget endpoints per IP
address, and everything here comes from one address, so reloading and resetting a
handful of times earns a `429` after which **the chat bubble simply stops
appearing, with nothing in the interface to say why**.

Then bring it up. The first command takes a few minutes; it is building the
database.

```bash
docker compose run --rm rails bundle exec rails db:chatwoot_prepare
docker compose up -d
```

Chatwoot is then on <http://localhost:3000>, but not immediately: Rails takes a
minute or so after the containers report as up. Wait for the page to actually
load before the next step, which needs a running Rails to talk to.

### 2. Set up the account and inboxes

```bash
cd ..
uv run python scripts/setup_chatwoot.py
```

This creates the account, an admin user, an access token, and the **two** inboxes
this needs, then prints the values for your `.env`. Safe to run again.

Two inboxes, because Chatwoot rejects incoming messages on a widget inbox with
*"Incoming messages are only allowed in Api inboxes"*. So there is a **Website**
inbox for real visitors and a **Programmatic** one for anything injecting
messages, including the check scripts.

It also applies two settings that are not discoverable and cost an evening
between them: agent auto-offline is turned off, or the widget tells every visitor
*"We are away at the moment"*; and the widget's email-collect prompt is turned
off, because it interrupts each new conversation before the agent has answered.

Credentials are written to `deploy/admin.local.txt`, which is gitignored. The
other scripts read it, so mostly you can leave it alone.

Open it for one thing: the email and password log you into
<http://localhost:3000> as an agent. Worth doing before demonstrating any of
this, because the dashboard is the other half of the story. Conversations appear
there as they happen, the agent's replies arrive in them, and a held refund shows
up as an open conversation with a private note explaining why a person is needed.

### 3. Configure the agent

```bash
uv sync
cp .env.example .env
```

`.env.example` is commented throughout and already carries working defaults, so
only four values are blank. Two of them step 2 has just printed for you:

```
CHATWOOT_TOKEN=<from step 2>
CHATWOOT_WIDGET_TOKEN=<from step 2>
OPENROUTER_API_KEY=<your model key, despite the name any provider works>
STRIPE_API_KEY=rk_test_<restricted key, see below>
```

Check `CHATWOOT_ACCOUNT_ID` against step 2 as well, though it is prefilled with
`1` and that is nearly always right. A fifth, `STRIPE_SETUP_KEY`, is needed only
to run the live checks.

**The Stripe key.** In a Stripe **sandbox**, create a restricted key with
**Charges and refunds: write** and **Customers: read**, everything else `None`.
Stripe has no standalone refunds permission, so write on "Charges and refunds" is
the narrowest grant that can issue one. The agent refuses to start if the key is
not a test key.

`STRIPE_SETUP_KEY` is your sandbox's ordinary secret key. It is only used by the
check scripts, to create the customers and charges a refund test needs — which
the agent's own key deliberately cannot do.

### 4. Add documentation to answer from

```bash
uv run python scripts/seed_helpcenter.py
```

Sixteen articles for a fictional coffee subscription, in Chatwoot's own help
centre. Invented on purpose, so nothing in the demo can be mistaken for a real
company's policy. Point `HELP_CENTRE_PORTAL` at a different portal to use real
documentation instead; no code changes.

### 5. Run it

Two processes, in two terminals:

```bash
uv run python agent.py         # the agent
uv run python demo/serve.py    # the demo page, on http://localhost:8080
```

Open <http://localhost:8080>, click the chat bubble, and ask something:

- *"can I skip a month?"* — answered from the help centre
- *"do you have decaf?"* — answered from the help centre
- *"how do I get a refund?"* — explains the process, and does **not** refund you
- *"I need a refund"* — actually refunds, in Stripe test mode

Replies take one to three seconds.

**Start a fresh conversation** on the page resolves the open queue, creates a new
customer with a refundable payment, and reloads. Use it between runs; a plain
refresh resumes the same conversation.

---

## The files

```
agent.py                  the whole agent: the graph, the policy, and the clients
knowledge.md              how to behave. No facts, no prices, no policies
demo/
  index.html              a page with the chat widget on it
  serve.py                serves that page and backs its reset button
deploy/
  docker-compose.yaml     Chatwoot, plus a mail server for the email channel
  env.example             upstream Chatwoot settings
scripts/
  setup_chatwoot.py       creates the account, inboxes and token
  seed_helpcenter.py      writes the mock documentation into Chatwoot
  reset_demo.py           clears conversations and payment fixtures
  live_check.py           the inbox adapter against a running Chatwoot
  e2e_check.py            everything live: model, Stripe test mode, Chatwoot
  reload_check.py         proves editing knowledge.md needs no restart
  classify_eval.py        labelled messages, scoring the refund judgement
tests/                    183 tests, no network and no credentials needed
notes/                    answers to questions this raised, written up as they came
```

`notes/` is the reasoning that did not belong in a docstring: how Chatwoot fits
against an existing system, how the email channel is wired, what running this on
a VPS costs, and what a student actually needs before session one.

### agent.py

One file, read top to bottom:

| Section | What lives there |
|---|---|
| Configuration | the `.env` reader, and the refund numbers as named constants |
| Types | `Message`, `Conversation`, `Proposal`, `Charge`, `Decision` |
| Ports | what an inbox, a payment provider and a model must offer |
| The refund rules | `refund_decision`. A pure function: no I/O, no model |
| Ours to touch | who spoke last, and what has been handled already |
| The graph | `State`, the nodes, the edges, and the six-line walk |
| Handling one conversation | builds the state and walks the graph once |
| The loop | poll, handle, sleep |
| Adapters | Chatwoot, the help centre, Stripe, the model |
| Entry point | wires the real adapters together from `.env` |

The decisions come first and the things that talk to the network come last.
Everything the agent uses is passed in rather than reached for, which is why the
tests need no network and no credentials.

### knowledge.md and the help centre

Two inputs, deliberately kept apart:

- **`knowledge.md` is how to behave.** Tone, when to ask a follow-up rather than
  hand over, and the standing rule never to invent. It states no facts at all,
  not even where a setting lives. It is re-read on every poll, so editing it
  changes behaviour within a couple of seconds, with no restart.
- **The help centre is what may be stated as fact.** Articles are fetched from
  Chatwoot, cached, and the few that match go into the prompt. If nothing
  matches, the agent is told to answer generally and state nothing specific about
  the product.

The split is the point. The model already knows how to write a support reply.
What it cannot know is your prices and policies. So it brings the language and
your documentation brings the facts, and a question with no matching article gets
a refusal rather than an invention.

---

## The refund policy

Auto-approval needs **all** of:

| Condition | Default |
|---|---|
| Amount at or under | `MAX_AUTO_REFUND_CENTS` = $50.00 |
| Purchase within | `MAX_CHARGE_AGE_DAYS` = 30 days |
| Previous refunds on the account | none |
| Rubric score at or above | `MIN_CONFIDENCE` = 0.6, two of three points |
| Charge not already refunded, in full or in part | — |
| Exactly one refundable charge on the account, not none and not several | — |

Anything else tells the customer a colleague will follow up, posts a private note
with the reason, leaves the conversation open, and touches no money.

**Confidence is scored, not self-reported.** The model is never asked how sure it
is. It answers three plain questions about the message and the score is
arithmetic in code:

| Signal | Question |
|---|---|
| `clear_request` | did they plainly ask, rather than wonder aloud? |
| `charge_identified` | did they say which payment, by date, amount or description? |
| `hedging` | did they hedge, with "maybe", "I think", "would I be able to"? |

A model rating its own certainty is introspecting, which it is bad at. These are
observable. They also make the note to a colleague specific: *"hedging: yes"*
rather than *"confidence 0.67"*.

Two of the three points are required. A plain *"I want my money back"* scores
clear request and no hedging but names no charge, so it lands on 0.67 and is
approved. *"Maybe I could get a refund?"* loses the hedging point too and goes to
a person. Requiring all three would hold nearly every genuine request, because
customers hardly ever name the payment.

Two details about Stripe that are easy to get wrong: it sets `refunded` only when
a charge is refunded *in full*, so a partly refunded charge looks untouched and
both count as refunded here; and customers rarely say which payment they mean, so
if more than one charge could still be refunded the agent refuses to choose.

---

## Not acting twice on the same message

Polling means seeing the same conversation repeatedly, so three things stop the
agent talking to itself:

- **Whoever spoke last decides.** If the newest message is ours, there is nothing
  to do. Chatwoot's own entries are excluded: activity records like labels, and
  the template messages the widget inserts, because nobody said them.
- **The last handled message id is recorded on the Chatwoot conversation**, in
  its `custom_attributes`, which the list response already returns. Chatwoot owns
  the memory, so it survives a restart and is shared with any other copy of the
  agent. An in-process set is kept too, purely as the cheap first check.
- **Refunds carry a Stripe idempotency key** built from the conversation and the
  charge. If the same refund ever reaches Stripe twice, it returns the original
  rather than issuing a second. This is the backstop that survives a crash
  between issuing a refund and telling the customer.

The poll costs **one request**, not one per conversation: the list response
already carries the newest message. That matters because refunds awaiting a human
stay open by design, so without it the agent would get slower the further behind
the humans were.

---

## Checking it works

```bash
uv run pytest                          # 183 tests, no network, no credentials
uv run python scripts/live_check.py    # the inbox adapter against live Chatwoot
uv run python scripts/e2e_check.py     # everything live, including a real refund
uv run python scripts/reload_check.py  # editing knowledge.md with the agent running
uv run python scripts/classify_eval.py # labelled messages through the refund judgement
```

`pytest` fakes Chatwoot, Stripe and the model, so the policy, the graph, the
adapters and the loop are all checked in isolation.

The live scripts need Chatwoot running and a filled-in `.env`. **Stop the agent
before `e2e_check.py` and `live_check.py`.** Both drive conversations themselves,
so a polling agent handles the same ones in parallel and the assertions fail for
the wrong reason: a reply arrives that the script did not send, and the failure
looks like a bug in the agent rather than two processes doing one job. Both
refuse to run if they detect a polling agent, so this is a message rather than a
mystery. `reload_check.py` needs no such care: it starts and stops its own agent,
since the thing it proves is that editing `knowledge.md` mid-run takes effect
without a restart.

`classify_eval.py` is the regression test for the judgement that decides whether
code may move money. That behaviour lives in a prompt, not a branch, so unit
tests cannot cover it. It reports failures by direction, because they are not
equally bad: answering a refund request with a reply is an annoyance, refunding
somebody who only asked a question is a real problem.

---

## Changing it

| To change | Edit |
|---|---|
| Tone, when to ask a follow-up | `knowledge.md` — takes effect within seconds |
| The facts it may state | Articles in your Chatwoot help centre |
| Refund thresholds | The constants at the top of `agent.py` |
| The model or provider | `OPENROUTER_MODEL`, `OPENROUTER_BASE_URL` in `.env` |
| How fast it replies | `POLL_INTERVAL_SECONDS` in `.env`, default 5 |

### Using a different model

The base URL accepts any OpenAI-shaped endpoint, so switching provider is
configuration rather than code. Groq instead of OpenRouter, for instance:

```
OPENROUTER_BASE_URL=https://api.groq.com/openai/v1
OPENROUTER_MODEL=openai/gpt-oss-20b
```

The `OPENROUTER_` prefix is a leftover from the first provider and is now a poor
name, since nothing here is specific to OpenRouter. Every one of these also
answers to `MODEL_API_KEY`, `MODEL_NAME`, `MODEL_BASE_URL` and `MODEL_PROVIDER`,
which are the honest names; the old ones keep working so existing `.env` files
do not break.

Two things worth knowing about the default. OpenRouter's `:free` models allow
**50 requests a day** on an account that has never bought credit, and the same
model can be served by several providers, only some of which honour a strict JSON
schema — set `MODEL_PROVIDER=Groq,Fireworks` to pin one. Neither matters in a
workshop, where everyone has their own key and sends a handful of messages, but
both will bite a machine doing repeated demo runs.

### The email channel

The widget is not the only way in. Chatwoot can log into a mailbox over IMAP and
turn each thread into a conversation, and the agent cannot tell the difference:
an email conversation reaches it looking exactly like a widget one.

So that this needs nobody's real mailbox, `deploy/docker-compose.yaml` runs
[GreenMail](https://greenmail-mail-test.github.io/greenmail/), a real IMAP and
SMTP server with two accounts already created. Add an Email inbox in Chatwoot
with:

| | |
|---|---|
| IMAP host / port | `greenmail` / `3143` |
| SMTP host / port | `greenmail` / `3025` |
| Login / password | `support` / `supportpass` |
| Address | `support@chatwoot.test` |

Three things will waste an hour otherwise:

- set the inbox's authentication to **`login`**, not the default `plain`, which
  many servers do not advertise
- the IMAP login is the mailbox user, `support`, not the full address
- **test messages need a `Message-ID` header** — Chatwoot silently discards mail
  without one, with no error anywhere

Both ports are published on `127.0.0.1` as well, so a script on the host can post
a test message without going through a mail client.

One caveat before demonstrating this on stage: the widget is instant, email is
not. Chatwoot fetches mail on a schedule, so a message takes a few minutes to
become a conversation, and only then does the agent's next poll see it.

---

## Provenance

Written as workshop material. The mock company, its help centre and every policy
in it are invented. Chatwoot is licensed separately by its authors.
