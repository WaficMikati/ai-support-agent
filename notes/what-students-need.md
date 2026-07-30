**What a student needs**

1. **Chatwoot, self-hosted.** Two ways, both with no licence cost and every channel included:
   - their own machine with Docker, which wants about 4GB of RAM
   - a VPS, roughly $9 a month on Hetzner or $24 on DigitalOcean or Vultr, the latter two having São Paulo and Mexico City regions

   On a VPS they also need a domain with HTTPS in front of it. Chatwoot Cloud's free plan does not cover this: live chat only, no email inbox and no API inbox.

2. **A mailbox for support.** On their own machine that means IMAP with an app password, plus SMTP for sending replies. On a VPS with a domain they can instead forward `support@` straight to Chatwoot, which is less fiddly than IMAP. Either way sending goes through an SMTP provider, since most hosts block the standard mail port.

3. **Their own free Groq account**, no card needed. The rate limits are per account, so it has to be theirs rather than a shared key.

4. **A Stripe sandbox and a restricted key**, for the refund half only.

5. **Python, and Cursor** to write the agent.

The only item on that list that costs anything is the VPS, and only if they pick one over running it on their own machine.
