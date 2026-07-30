"""Give somebody else an administrator login for this Chatwoot.

    uv run python scripts/add_admin.py alejandro@example.com "Alejandro Sanchez"

Prints a password, and writes nothing down. Run it again for the same address
and it resets the password rather than failing, which is the usual reason to
run it twice.

Their own login rather than a shared one, so the dashboard shows who replied to
what, and so revoking one person does not lock everybody out. Both see the same
conversations, notes and contacts: an administrator on this account can read all
of it, and this is a demo on a laptop, not a place to keep anything private.

Goes through `rails runner` for the same reason the first-time setup does: the
API needs a token belonging to a user, and creating the first one is the thing
being asked for.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEPLOY = ROOT / "deploy"


RUBY = '''
require "securerandom"
password = "Aa1!" + SecureRandom.hex(12)
account = Account.first
raise "no account yet, run scripts/setup_chatwoot.py first" if account.nil?

user = User.find_by(email: {email!r})
if user.nil?
  user = User.new(
    name: {name!r}, email: {email!r},
    password: password, password_confirmation: password
  )
  user.skip_confirmation! if user.respond_to?(:skip_confirmation!)
  user.save!
  created = true
else
  # Running it again is almost always somebody who has lost the password.
  user.password = password
  user.password_confirmation = password
  user.save!
  created = false
end

membership = AccountUser.find_or_initialize_by(account_id: account.id, user_id: user.id)
membership.role = 1  # administrator
# Without this the widget tells visitors "we are away" whenever the only agents
# are ones that never open a dashboard.
membership.auto_offline = false
membership.availability = :online
membership.save!

Inbox.where(account_id: account.id).find_each do |inbox|
  InboxMember.find_or_create_by!(inbox_id: inbox.id, user_id: user.id)
end

puts "RESULT|" + [created ? "created" : "password reset", password].join("|")
'''


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__.strip().splitlines()[2].strip())
        return 1
    email = sys.argv[1].strip()
    name = (sys.argv[2] if len(sys.argv) > 2 else email.split("@")[0]).strip()

    if not (DEPLOY / ".env").exists():
        print("deploy/.env is missing. Start Chatwoot first, see the README.")
        return 1

    result = subprocess.run(
        [
            "docker", "compose", "exec", "-T", "rails",
            "bundle", "exec", "rails", "runner",
            RUBY.format(email=email, name=name),
        ],
        cwd=DEPLOY, capture_output=True, text=True, timeout=300,
    )
    line = next(
        (row for row in result.stdout.splitlines() if row.startswith("RESULT|")), None
    )
    if line is None:
        print("failed. Is Chatwoot up? `cd deploy && docker compose ps`\n")
        print((result.stderr or result.stdout)[-800:])
        return 1

    _, what, password = line.split("|")
    print(f"\n  {what}: {email}")
    print(f"  password: {password}")
    print("\nThey sign in at whichever address reaches this Chatwoot:")
    print("  http://localhost:3000/app/login   from this machine")
    print("  the tunnel URL                    from anywhere else")
    print("\nThe password is shown once and not stored. Run this again to reset it.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
