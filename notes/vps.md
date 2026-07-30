**On a VPS**

Yes, and that is the natural shape for it. Self-hosted Chatwoot is built to run on a server, and the agent is a plain Python process that runs anywhere. Nothing we have written would change: the agent just points at a different URL.

On the plans question, the important bit is that **self-hosting has no plan at all.** The free/paid tiers only exist on Chatwoot Cloud. When you run it yourself you get every channel, unlimited agents and unlimited conversations, for no licence cost, whether that is on a laptop or a server. So instead of paying per agent per month, you pay for the box.

Roughly, for the 4GB / 2 vCPU that Chatwoot wants:

- Hetzner, about $8–9 a month
- DigitalOcean or Vultr, about $24 a month

Vultr and DigitalOcean have Latin American regions, São Paulo and Mexico City among them; Hetzner is Europe and the US only, so it is cheaper but further away.

The other free pieces are unaffected, because they have nothing to do with where the server is. Groq's free tier is per account. Stripe test mode is free.

What a server actually buys us, beyond being always on:

- **Email by forwarding becomes possible.** That was the option ruled out on a laptop, because it needs a publicly reachable address. On a VPS with a domain, we are no longer limited to IMAP.
- **The chat widget becomes reachable by real visitors**, which it simply is not right now.

Two things it adds to the setup: a domain with HTTPS in front of it, which is a reverse proxy and a certificate, and an SMTP provider for sending, because most hosts block the standard mail port outright.
