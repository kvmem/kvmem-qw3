"""Private site state operations for the host initializer/grader only."""
import json
from datetime import datetime
import math
from pathlib import Path
import re
import time
import urllib.request

from .source import ROOT

PORTS = {"formcraft":18110, "teamchat":18111, "cloudcrm":18112, "calendar":18113, "mailhub":18114}


class Site:
    def __init__(self, app, user, claims=None):
        if app not in PORTS or not re.fullmatch(r"[a-zA-Z0-9_-]{8,80}", user):
            raise ValueError("Unknown private site or invalid test identity")
        self.app, self.user = app, user
        self.claims = Path(claims) if claims is not None else ROOT / "apps/computer_use/.runtime/osworld-site-identities"
        self.origin = f"http://127.0.0.1:{PORTS[app]}"
        self.opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    @property
    def browser_url(self):
        return f"http://{self.app}.qw3.test:18090/?cookie={self.user}"

    def request(self, method, path, body=None):
        request = urllib.request.Request(self.origin+path, method=method,
            data=None if body is None else json.dumps(body).encode(),
            headers={"Content-Type":"application/json", "Cookie":"user_id="+self.user})
        with self.opener.open(request, timeout=20) as response:
            return json.load(response)

    def state(self):
        response = self.request("GET", "/api/state")
        if response.get("user_id") != self.user or not isinstance(response.get("state"), dict):
            raise ValueError("Private site returned another identity or malformed state")
        return response["state"]

    def initialize(self, state, *, consumed_fields=()):
        if consumed_fields and (self.app not in {'teamchat','mailhub'} or set(consumed_fields)-{'time_data','action_data','enable_notifications'}):
            raise ValueError('Unknown scheduler-consumed initializer fields')
        # GET /state creates nonempty demo records on most upstream sites.
        # Claim locally before that first GET, then prove the backend created
        # this identity during this request. These containers share the host
        # clock; Calendar timestamps have millisecond precision. Refuse old
        # identities, concurrent initializers, and retries after uncertainty.
        self.claims.mkdir(parents=True, exist_ok=True)
        claim = self.claims / (self.app + "-" + self.user + ".json")
        with claim.open("x") as handle:
            json.dump({"app": self.app, "user": self.user, "status": "claimed", "at": time.time()}, handle)
        started = math.floor(time.time() * 1000) / 1000
        before = self.state()
        ended = time.time()
        meta = before.get("meta", {})
        try:
            created = datetime.fromisoformat(meta["created_at"].replace("Z", "+00:00")).timestamp()
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("Cannot establish a newly created backend identity") from exc
        if not (started <= created <= ended and meta.get("version") == 1 and before.get("note") is None):
            raise ValueError("Attempt identity already exists or backend clock differs; use a fresh identity")
        (self.claims / (self.app + "-" + self.user + "-before.json")).write_text(json.dumps(before, indent=2))
        response = self.request("PUT", "/api/state", state)
        if response.get("user_id") != self.user:
            raise ValueError("Private site initializer identity mismatch")
        actual = self.state()
        expected = state.get("data")
        if consumed_fields:
            expected = {k:v for k,v in expected.items() if k not in consumed_fields}
            # These upstream backends move dynamic fields into the scheduler.
            # Retain the exact schedule outside agent input for delivery audits.
            (self.claims / (self.app + '-' + self.user + '-scheduled-input.json')).write_text(json.dumps(state,indent=2))
        if actual.get("data") != expected:
            raise ValueError("Private site did not preserve initialized task data")
        claim.write_text(json.dumps({"app": self.app, "user": self.user, "status": "initialized", "at": time.time()}, indent=2))
        return actual
