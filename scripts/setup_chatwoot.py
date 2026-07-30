"""Prepare a fresh Chatwoot for this agent, and write down what it created.

    uv run python scripts/setup_chatwoot.py

Creates the account, an admin user, an access token, and the two inboxes this
needs, then applies the settings that are not obvious and cost an evening to
find. Safe to re-run: everything is looked up before it is created, so running
it against a Chatwoot that is already set up just reports what is there.

Goes through `rails runner` rather than the API because of a chicken and egg
problem: the API needs an access token, and the token belongs to a user that
does not exist yet.

Writes deploy/admin.local.txt with the token and ids. That file is gitignored
and the other scripts read it.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEPLOY = ROOT / "deploy"

ACCOUNT_NAME = "Support demo"
ADMIN_EMAIL = "ops@example.com"
WIDGET_INBOX = "Website"
API_INBOX = "Programmatic"

RUBY = f'''
require "securerandom"
password = "Aa1!" + SecureRandom.hex(12)

# The first account, whatever it is called, rather than one matching a name.
# Keying on the name created a second account on a Chatwoot that was already set
# up, because somebody had renamed the first one.
account = Account.first || Account.create!(name: {ACCOUNT_NAME!r})

user = User.find_by(email: {ADMIN_EMAIL!r})
if user.nil?
  user = User.new(
    name: "Ops", email: {ADMIN_EMAIL!r},
    password: password, password_confirmation: password
  )
  user.skip_confirmation! if user.respond_to?(:skip_confirmation!)
  user.save!
else
  password = "(unchanged, see deploy/admin.local.txt from the first run)"
end

membership = AccountUser.find_or_initialize_by(account_id: account.id, user_id: user.id)
membership.role = 1
# Chatwoot decides "online or away" from human agent presence, meaning a live
# dashboard websocket. This agent authenticates with a token and never opens
# one, so without this the widget greets visitors with "We are away at the
# moment" on behalf of something that answers in seconds.
membership.auto_offline = false
membership.availability = :online
membership.save!

widget_channel = Channel::WebWidget.find_or_create_by!(
  account_id: account.id, website_url: "http://localhost:8080"
)
widget = Inbox.find_or_initialize_by(
  account_id: account.id, channel_id: widget_channel.id, channel_type: "Channel::WebWidget"
)
widget.name = {WIDGET_INBOX!r} if widget.name.blank?
# Without this the widget interrupts every new conversation to ask for an email
# before the agent has answered. The demo page identifies the visitor instead,
# the way a real site does for a signed-in customer.
widget.enable_email_collect = false
widget.save!

# A second inbox, because Chatwoot refuses incoming messages on a widget inbox:
# "Incoming messages are only allowed in Api inboxes". Anything that injects a
# message programmatically, including the check scripts, needs this one.
api_channel = Channel::Api.find_by(account_id: account.id)
if api_channel.nil?
  api_channel = Channel::Api.new(account: account, identifier: SecureRandom.hex(12))
  api_channel.save!
end
api = Inbox.find_or_initialize_by(
  account_id: account.id, channel_id: api_channel.id, channel_type: "Channel::Api"
)
api.name = {API_INBOX!r} if api.name.blank?
api.save!

[widget, api].each do |inbox|
  InboxMember.find_or_create_by!(inbox_id: inbox.id, user_id: user.id)
end

token = user.access_token&.token || AccessToken.create!(owner: user).token

puts "RESULT|" + [
  account.id, token, password,
  widget.id, widget_channel.website_token,
  api.id, api_channel.identifier,
].join("|")
'''


def main() -> int:
    if not (DEPLOY / ".env").exists():
        print("deploy/.env is missing. Start Chatwoot first:")
        print("  cd deploy && cp env.example .env   # then set the three secrets")
        print("  docker compose run --rm rails bundle exec rails db:chatwoot_prepare")
        print("  docker compose up -d")
        return 1

    print("asking Chatwoot to create what is missing (this takes a few seconds)")
    result = subprocess.run(
        ["docker", "compose", "exec", "-T", "rails", "bundle", "exec", "rails", "runner", RUBY],
        cwd=DEPLOY, capture_output=True, text=True, timeout=300,
    )
    line = next(
        (row for row in result.stdout.splitlines() if row.startswith("RESULT|")), None
    )
    if line is None:
        print("failed. Is Chatwoot up? `cd deploy && docker compose ps`\n")
        print((result.stderr or result.stdout)[-800:])
        return 1

    (
        _,
        account_id,
        token,
        password,
        widget_inbox,
        widget_token,
        api_inbox,
        api_identifier,
    ) = line.split("|")

    target = DEPLOY / "admin.local.txt"
    target.write_text(
        "\n".join(
            [
                "chatwoot_url=http://localhost:3000",
                f"email={ADMIN_EMAIL}",
                f"password={password}",
                f"account_id={account_id}",
                f"access_token={token}",
                f"widget_inbox_id={widget_inbox}",
                f"widget_token={widget_token}",
                f"api_inbox_id={api_inbox}",
                f"api_identifier={api_identifier}",
            ]
        )
        + "\n"
    )
    target.chmod(0o600)

    print(f"\n  account {account_id}")
    print(f"  inboxes: {widget_inbox} (web widget), {api_inbox} (api)")
    print(f"  credentials written to {target.relative_to(ROOT)}")
    print("\nNow put these in .env:")
    print(f"  CHATWOOT_URL=http://localhost:3000")
    print(f"  CHATWOOT_ACCOUNT_ID={account_id}")
    print(f"  CHATWOOT_TOKEN={token}")
    print(f"  CHATWOOT_WIDGET_TOKEN={widget_token}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
