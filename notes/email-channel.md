**Connecting email**

Chatwoot supports two inbound routes, and only one of them fits how we are running this.

**IMAP — the one to use.** Chatwoot logs into a real mailbox and pulls messages out of it. One inbox gets an address plus IMAP host, port, login and password, and SMTP details for sending the replies. Nothing needs to be publicly reachable, so it works on a laptop with no hosting and no DNS.

**Forwarding — not viable yet.** Chatwoot can also accept mail forwarded to an address it owns, but on a self-hosted install that runs through Rails Action Mailbox: it needs a public ingress URL, an inbound mail domain, and a provider such as Mailgun or SES with a signing key. That is real infrastructure and a production decision, not something to set up for a workshop.

So the practical path is a mailbox we control with IMAP switched on. With Gmail that means two-factor plus an app password; any provider with IMAP works the same way. Outbound replies use SMTP on the same inbox, and no SMTP credentials are configured at the moment, so that has to be filled in before a reply can actually leave.

One timing difference worth knowing before demoing it. The widget is instant. Email is not: Chatwoot fetches mail on a schedule, so a message takes a few minutes to show up as a conversation, and the agent then picks it up on its next poll. Fine for support, just not dramatic on stage.

The agent itself needs no changes. An email conversation reaches it looking exactly like a widget one.
