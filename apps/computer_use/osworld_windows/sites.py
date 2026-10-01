"""Private site state operations for the host initializer/grader only."""
import json
import re
import urllib.request

PORTS = {"formcraft":18110, "teamchat":18111, "cloudcrm":18112, "calendar":18113, "mailhub":18114}


class Site:
    def __init__(self, app, user):
        if app not in PORTS or not re.fullmatch(r"[a-zA-Z0-9_-]{8,80}", user):
            raise ValueError("Unknown private site or invalid test identity")
        self.app, self.user = app, user
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

    def initialize(self, state):
        # Fresh user identity per attempt, even when reusing a VM snapshot.
        if self.state().get("data"):
            raise ValueError("Attempt identity already has task data; use a fresh identity")
        response = self.request("PUT", "/api/state", state)
        if response.get("user_id") != self.user:
            raise ValueError("Private site initializer identity mismatch")
        actual = self.state()
        if actual.get("data") != state.get("data"):
            raise ValueError("Private site did not preserve initialized task data")
        return actual
