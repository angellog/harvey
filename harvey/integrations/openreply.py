"""OpenReply bridge — Harvey's link to the cloud control plane.

Harvey runs on the operator's own machine, where the Claude CLI is logged in.
OpenReply (Next.js on Vercel, Postgres on Railway) is reachable from a phone.
Neither can open a socket to the other reliably, so the traffic is one-way:
Harvey pushes what it learned and polls for what the operator decided. There is
no inbound port, no tunnel, and nothing to expose on the operator's network.

Every request carries three headers:

    X-Harvey-Agent:     which Harvey instance is calling
    X-Harvey-Timestamp: unix seconds, inside the signed material
    X-Harvey-Signature: sha256=<hex> over "<timestamp>.<raw body>"

The timestamp is signed rather than merely sent, so moving the clock forward
invalidates the signature instead of extending a captured request's life.
Keep this in step with lib/harvey/signature.ts on the OpenReply side.
"""

import hashlib
import hmac
import json
import logging
import time

logger = logging.getLogger("harvey.openreply")

AGENT_HEADER = "X-Harvey-Agent"
TIMESTAMP_HEADER = "X-Harvey-Timestamp"
SIGNATURE_HEADER = "X-Harvey-Signature"

DEFAULT_TIMEOUT_SECONDS = 30
# One sync must never grow unbounded: a first run after a big discover could
# otherwise try to push thousands of rows in a single request body.
MAX_BATCH = 200


class OpenReplyError(Exception):
    """The bridge could not complete a call. Harvey keeps its own state and
    retries on the next heartbeat; nothing is dropped because of this."""


def canonical_body(payload: dict) -> str:
    """The exact bytes that get signed and sent.

    Signing happens over a string, not over a dict, because any re-encoding
    between signing and sending would break the signature. Callers must send
    this return value verbatim as the request body.
    """
    return json.dumps(payload, separators=(",", ":"), sort_keys=True)


def sign_payload(raw_body: str, timestamp: str, secret: str) -> str:
    digest = hmac.new(
        secret.encode("utf-8"),
        f"{timestamp}.{raw_body}".encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()
    return f"sha256={digest}"


def build_headers(raw_body: str, agent_id: str, secret: str, timestamp: str | None = None) -> dict:
    ts = timestamp or str(int(time.time()))
    return {
        "Content-Type": "application/json",
        AGENT_HEADER: agent_id,
        TIMESTAMP_HEADER: ts,
        SIGNATURE_HEADER: sign_payload(raw_body, ts, secret),
    }


class OpenReplyClient:
    """Thin signed-HTTP client. Holds no state of its own: Harvey's SQLite
    stays the source of truth for Harvey, and OpenReply's Postgres stays the
    source of truth for the operator's decisions."""

    def __init__(self, base_url: str, agent_id: str, secret: str, timeout: float = DEFAULT_TIMEOUT_SECONDS):
        self.base_url = (base_url or "").rstrip("/")
        self.agent_id = (agent_id or "").strip()
        self.secret = (secret or "").strip()
        self.timeout = timeout

    def is_configured(self) -> bool:
        """Whether the bridge can be used at all (no network)."""
        return bool(self.base_url and self.agent_id and self.secret)

    async def post(self, path: str, payload: dict) -> dict:
        if not self.is_configured():
            raise OpenReplyError("OpenReply bridge is not configured")

        # Imported here, not at module scope, so the signing helpers above stay
        # importable (and testable) without the HTTP dependency present.
        import httpx

        raw = canonical_body(payload)
        url = f"{self.base_url}{path}"
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    url,
                    content=raw.encode("utf-8"),
                    headers=build_headers(raw, self.agent_id, self.secret),
                )
        except Exception as exc:  # network, DNS, TLS, timeout
            raise OpenReplyError(f"{path}: {type(exc).__name__}: {exc}") from exc

        if response.status_code >= 400:
            raise OpenReplyError(
                f"{path}: HTTP {response.status_code}: {response.text[:200]}"
            )
        try:
            return response.json()
        except ValueError as exc:
            raise OpenReplyError(f"{path}: response was not JSON") from exc


# ─────────────────────────────────────────────────────────────────────────────
# Body rewriting
#
# Harvey's Writer never knows the tracked URL: the link does not exist until
# OpenReply mints it, which happens after the email is written. So the bridge
# splices it in on the way out.
#
# Two rules from Harvey's own side shape this, and both are load-bearing:
# prompts/writer.md says "No links in email 1. At most one link total across
# the sequence", and harvey/gate.py enforces MAX_LINKS = 1 at send time. So a
# link is only ever added from step 2 onwards, and only one is ever added.
# ─────────────────────────────────────────────────────────────────────────────

LINK_MODES = ("replace", "append", "none")
LINK_DESTINATIONS = ("whatsapp", "default")

DEFAULT_LINK_CTA = "If WhatsApp is easier for you:"


def body_has_tracked_link(body: str, base_url: str) -> bool:
    """Whether this body already carries a link from this control plane.

    The idempotency guard for rewriting: a retried mint must not append the
    same call-to-action twice.
    """
    if not base_url:
        return False
    return f"{base_url.rstrip('/')}/r/" in (body or "")


def rewrite_body(
    body: str,
    tracked_url: str,
    *,
    mode: str = "append",
    placeholder_url: str = "",
    cta: str = DEFAULT_LINK_CTA,
) -> str:
    """Put ``tracked_url`` into ``body``, or return it unchanged.

    ``replace`` swaps a URL the Writer was told to include (a booking page, for
    advisory outreach) for the tracked one, which keeps the email at exactly one
    link. It falls back to appending when that URL is absent, because a reply
    the Writer phrased without the link should still be trackable.

    ``append`` adds one short line. ``none`` never touches the body.
    """
    if mode not in LINK_MODES:
        raise ValueError(f"link mode must be one of {LINK_MODES}, got {mode!r}")
    if mode == "none" or not tracked_url:
        return body

    if mode == "replace" and placeholder_url and placeholder_url in body:
        return body.replace(placeholder_url, tracked_url)

    line = f"{cta.strip()} {tracked_url}".strip()
    return f"{(body or '').rstrip()}\n\n{line}\n"


def should_mint_link(
    item: dict,
    *,
    base_url: str,
    link_from_step: int = 2,
    link_in_replies: bool = False,
) -> bool:
    """Whether this outbox row should get a tracked link.

    Step 1 is excluded on purpose. A cold first touch that asks a question
    outperforms one that asks for a click, and Harvey's writer prompt forbids
    it outright.
    """
    if item.get("status") not in ("pending_review", "approved"):
        return False
    if body_has_tracked_link(item.get("body", ""), base_url):
        return False

    kind = item.get("kind", "sequence")
    if kind == "reply":
        return link_in_replies
    if kind != "sequence":
        return False
    try:
        return int(item.get("step", 1)) >= link_from_step
    except (TypeError, ValueError):
        return False


# ─────────────────────────────────────────────────────────────────────────────
# Command application
#
# The operator's decisions arrive as commands, each delivered at least once.
# Every handler below must therefore be safe to run twice, and must never
# widen what Harvey was already going to do: a command can approve an email
# that is waiting, or stop one, but it can never conjure a send that Harvey
# did not write.
# ─────────────────────────────────────────────────────────────────────────────

# Harvey's own state transitions already refuse to re-approve a sent item
# (approve_outbox only matches status = 'pending_review'), so "already_sent"
# is a normal outcome rather than an error.
COMMAND_KINDS = (
    "APPROVE_EMAIL",
    "REJECT_EMAIL",
    "EDIT_EMAIL",
    "APPROVE_ALL",
    "PAUSE_SENDING",
    "RESUME_SENDING",
)

# Harvey's own kill switch, shared with harvey/agents/sender.py and
# harvey/cli.py. The value is the human-readable reason, which its CLI and
# local dashboard print back, so a pause from the phone has to say where it
# came from rather than storing a bare flag.
SENDING_PAUSED_KEY = "sending_paused"
SENDING_PAUSED_REASON = "paused from OpenReply"


def command_is_expired(command: dict, now_iso: str) -> bool:
    """Whether this command sat in the queue too long to apply.

    A laptop can be closed for days. An approval given on Monday should not
    fire on Thursday against a prospect whose situation has moved on, so the
    control plane stamps an expiry and Harvey honours it.
    """
    expires_at = command.get("expiresAt")
    if not expires_at:
        return False
    return str(expires_at) <= now_iso


class CommandApplier:
    """Applies one command to Harvey's own state.

    Deliberately not a method on the HTTP client: this touches SQLite only, so
    it is testable with a fake state manager and no network at all.
    """

    def __init__(self, state, logger_=None):
        self.state = state
        self.log = logger_ or logger

    async def apply(self, command: dict, now_iso: str) -> dict:
        """Returns {"commandId", "result"} for the ack. Never raises for a
        command Harvey simply cannot apply — that is a result, not a failure,
        and raising would stall every command behind it."""
        command_id = command.get("id", "")
        kind = command.get("kind", "")
        target = command.get("targetHarveyId") or ""

        if kind not in COMMAND_KINDS:
            # An unknown kind means the control plane is newer than this
            # Harvey. Ack it so it stops being redelivered, and say why.
            return {"commandId": command_id, "result": f"unsupported:{kind}"}

        if command_is_expired(command, now_iso):
            return {"commandId": command_id, "result": "expired"}

        try:
            result = await self._dispatch(kind, target, command.get("payload") or {})
        except Exception as exc:  # one bad command must not stall the queue
            self.log.warning("openreply: command %s (%s) failed: %s", command_id, kind, exc)
            return {"commandId": command_id, "result": f"error:{type(exc).__name__}"}

        return {"commandId": command_id, "result": result}

    async def _dispatch(self, kind: str, target: str, payload: dict) -> str:
        if kind == "APPROVE_EMAIL":
            return await self._approve(target)
        if kind == "REJECT_EMAIL":
            return await self._reject(target, payload.get("reason") or "rejected in OpenReply")
        if kind == "EDIT_EMAIL":
            return await self._edit(target, payload)
        if kind == "APPROVE_ALL":
            count = await self.state.approve_outbox()
            return f"approved_all:{count}"
        if kind == "PAUSE_SENDING":
            await self.state.set_setting(SENDING_PAUSED_KEY, SENDING_PAUSED_REASON)
            return "paused"
        if kind == "RESUME_SENDING":
            await self.state.set_setting(SENDING_PAUSED_KEY, "")
            return "resumed"
        return f"unsupported:{kind}"

    async def _approve(self, item_id: str) -> str:
        if not item_id:
            return "not_found"
        item = await self.state.get_outbox_item(item_id)
        if not item:
            return "not_found"
        # approve_outbox only moves pending_review rows, so this reports what
        # actually happened rather than what was asked for.
        changed = await self.state.approve_outbox(item_id)
        if changed:
            return "approved"
        return f"already_{item.get('status', 'unknown')}"

    async def _reject(self, item_id: str, reason: str) -> str:
        if not item_id:
            return "not_found"
        item = await self.state.get_outbox_item(item_id)
        if not item:
            return "not_found"
        if item.get("status") not in ("pending_review", "approved"):
            return f"already_{item.get('status', 'unknown')}"
        # Harvey has no "rejected" status; a rejection is a cancellation whose
        # reason is recorded, which keeps this mirror of its vocabulary honest.
        await self.state.update_outbox_item(item_id, status="cancelled", error=reason)
        return "rejected"

    async def _edit(self, item_id: str, payload: dict) -> str:
        """Applies a correction made from the phone.

        Only a waiting email can be edited: changing the text of something
        already sent would make the mirror lie about what the prospect read.
        """
        if not item_id:
            return "not_found"
        item = await self.state.get_outbox_item(item_id)
        if not item:
            return "not_found"
        if item.get("status") != "pending_review":
            return f"already_{item.get('status', 'unknown')}"

        fields = {}
        subject = (payload.get("subject") or "").strip()
        body = (payload.get("body") or "").strip()
        if subject:
            fields["subject"] = subject
        if body:
            fields["body"] = body
        if not fields:
            return "nothing_to_edit"

        await self.state.update_outbox_item(item_id, **fields)
        return "edited"


# ─────────────────────────────────────────────────────────────────────────────
# Payload builders
#
# Pure functions from Harvey's own row shapes to the wire shapes. Separated
# from the client so they can be tested without a database or a network, and
# so a schema change on either side fails in one obvious place.
# ─────────────────────────────────────────────────────────────────────────────

# Harvey's prospects table carries denormalised company columns; the companies
# table is the canonical record. Both are pushed so the control plane can group
# prospects by business, which is the question B2B outreach actually asks.


def company_payload(row: dict) -> dict:
    return {
        "harveyId": row.get("id", ""),
        "name": row.get("name") or "",
        "domain": row.get("domain") or None,
        "website": row.get("website") or None,
        "industry": row.get("industry") or None,
        "location": row.get("location") or None,
        "sourceUrl": row.get("source_url") or None,
    }


def prospect_payload(row: dict) -> dict:
    return {
        "harveyId": row.get("id", ""),
        "companyHarveyId": row.get("company_id") or None,
        "firstName": row.get("first_name") or None,
        "lastName": row.get("last_name") or None,
        "email": row.get("email") or "",
        # Only the four verdicts the control plane accepts; anything else is
        # dropped rather than rejecting the whole batch over one odd row.
        "emailStatus": (
            row.get("email_status")
            if row.get("email_status") in ("verified", "risky", "guess", "invalid")
            else None
        ),
        "title": row.get("title") or None,
        "status": row.get("status") or None,
        "score": int(row["score"]) if str(row.get("score") or "").isdigit() else None,
        "personalizationNotes": row.get("personalization_notes") or None,
    }


def email_payload(row: dict) -> dict:
    return {
        "harveyId": row.get("id", ""),
        "prospectHarveyId": row.get("prospect_id") or None,
        "campaignHarveyId": row.get("campaign_id") or None,
        "step": int(row.get("step") or 1),
        "kind": row.get("kind") or "sequence",
        "toEmail": row.get("to_email") or "",
        "subject": row.get("subject") or "",
        "body": row.get("body") or "",
        "status": row.get("status") or "pending_review",
        "sendAt": _as_wire_timestamp(row.get("send_at")),
        "sentAt": _as_wire_timestamp(row.get("sent_at")),
        "error": row.get("error") or None,
    }


def _as_wire_timestamp(value) -> str | None:
    """SQLite stores naive UTC strings; the wire wants an offset.

    Without the offset the control plane would read a send time as local to
    whatever region its server sits in, which is a silent several-hour error on
    every scheduled email.
    """
    if not value:
        return None
    text = str(value).strip().replace(" ", "T")
    if text.endswith("Z") or "+" in text[10:] or text[10:].count("-") > 0:
        return text
    return f"{text}+00:00"


# ─────────────────────────────────────────────────────────────────────────────
# The bridge loop
# ─────────────────────────────────────────────────────────────────────────────

# Cursor keys in Harvey's own settings table. Pushing is incremental: only
# rows whose updated_at moved since the last accepted sync go out, so a long
# running instance does not re-send its whole database every minute.
CURSOR_PROSPECTS = "openreply.cursor.prospects"
CURSOR_OUTBOX = "openreply.cursor.outbox"
LAST_OK_KEY = "openreply.last_ok"
LAST_ERROR_KEY = "openreply.last_error"

DEFAULT_SYNC_INTERVAL_SECONDS = 60
MIN_SYNC_INTERVAL_SECONDS = 15
# Consecutive failures back off to this, so a control plane that is down (or a
# laptop on a hotel connection) is retried patiently instead of hammered.
MAX_BACKOFF_SECONDS = 900


class OpenReplyBridge:
    """Keeps Harvey and its control plane in step.

    Deliberately boring: it reads rows, posts them, applies what comes back,
    and records where it got to. It makes no model calls — not one — so it
    costs nothing against the operator's Claude quota and can run during quiet
    hours, which is exactly when approvals get made.
    """

    def __init__(self, state, client: OpenReplyClient, config=None, logger_=None):
        self.state = state
        self.client = client
        self.config = config
        self.log = logger_ or logger
        self.applier = CommandApplier(state, self.log)
        self._consecutive_failures = 0
        # Held until the next cycle: a command cannot be acked in the same
        # request that delivered it. Per-instance, never class-level, or two
        # bridges in one process would ack each other's commands.
        self._pending_acks: list[dict] = []

    # ── configuration, with defaults that are safe when unset ──────────────

    @property
    def base_url(self) -> str:
        return self.client.base_url

    @property
    def sync_interval(self) -> int:
        value = getattr(self.config, "sync_interval_seconds", DEFAULT_SYNC_INTERVAL_SECONDS)
        return max(MIN_SYNC_INTERVAL_SECONDS, int(value or DEFAULT_SYNC_INTERVAL_SECONDS))

    @property
    def link_mode(self) -> str:
        mode = getattr(self.config, "link_mode", "append") or "append"
        return mode if mode in LINK_MODES else "append"

    @property
    def link_from_step(self) -> int:
        return int(getattr(self.config, "link_from_step", 2) or 2)

    @property
    def link_in_replies(self) -> bool:
        return bool(getattr(self.config, "link_in_replies", False))

    @property
    def link_destination(self) -> str:
        destination = getattr(self.config, "link_destination", "whatsapp") or "whatsapp"
        return destination if destination in LINK_DESTINATIONS else "whatsapp"

    @property
    def link_url(self) -> str:
        return getattr(self.config, "link_url", "") or ""

    @property
    def link_cta(self) -> str:
        return getattr(self.config, "link_cta", DEFAULT_LINK_CTA) or DEFAULT_LINK_CTA

    # ── one cycle ──────────────────────────────────────────────────────────

    async def sync_once(self) -> dict:
        """Mint any missing links, push what changed, apply what came back.

        Minting comes first so an email that gains a tracked link is pushed
        with the final body, not the draft the prospect will never read.
        """
        minted = await self._mint_links()
        pushed = await self._push()
        return {"minted": minted, **pushed}

    async def _mint_links(self) -> int:
        """Ask the control plane for one tracked link per eligible email, then
        splice it into the body.

        Skipped entirely in "none" mode, and never applied to step 1."""
        if self.link_mode == "none":
            return 0

        candidates: list[dict] = []
        for status in ("pending_review", "approved"):
            rows = await self.state.get_outbox(status=status, limit=MAX_BATCH)
            candidates.extend(
                row
                for row in rows
                if should_mint_link(
                    row,
                    base_url=self.base_url,
                    link_from_step=self.link_from_step,
                    link_in_replies=self.link_in_replies,
                )
            )

        minted = 0
        for row in candidates:
            try:
                response = await self.client.post(
                    "/api/harvey/v1/links",
                    {
                        "emailHarveyId": row["id"],
                        "destination": self.link_destination,
                        "label": f"{row.get('kind', 'sequence')} step {row.get('step', 1)}",
                    },
                )
            except OpenReplyError as exc:
                # A link that cannot be minted is not a reason to hold the
                # email hostage: it goes out without one on the next cycle's
                # attempt, or waits, but nothing is lost either way.
                self.log.warning("openreply: link mint failed for %s: %s", row["id"], exc)
                continue

            tracked_url = (response.get("data") or {}).get("url") or ""
            if not tracked_url:
                continue

            new_body = rewrite_body(
                row.get("body", ""),
                tracked_url,
                mode=self.link_mode,
                placeholder_url=self.link_url,
                cta=self.link_cta,
            )
            if new_body == row.get("body"):
                continue
            await self.state.update_outbox_item(row["id"], body=new_body)
            minted += 1

        return minted

    async def _push(self) -> dict:
        """Send what changed, ack what was applied, apply what comes back."""
        now_iso = _utcnow_iso()
        prospect_cursor = await self.state.get_setting(CURSOR_PROSPECTS)
        outbox_cursor = await self.state.get_setting(CURSOR_OUTBOX)

        prospects = await self._rows_since("prospects", prospect_cursor)
        emails = await self._rows_since("outbox", outbox_cursor)
        companies = await self._companies_for(prospects)

        payload = {
            "runtime": {
                "version": _harvey_version(),
                "brainProvider": getattr(
                    getattr(self.config, "_brain", None), "provider", ""
                )
                or "",
            },
            "acks": self._pending_acks,
            "companies": [company_payload(row) for row in companies],
            "prospects": [prospect_payload(row) for row in prospects],
            "emails": [email_payload(row) for row in emails],
        }

        try:
            response = await self.client.post("/api/harvey/v1/sync", payload)
        except OpenReplyError as exc:
            self._consecutive_failures += 1
            await self.state.set_setting(LAST_ERROR_KEY, f"{now_iso} {exc}")
            raise

        # Acks were accepted, so stop resending them.
        self._pending_acks = []
        self._consecutive_failures = 0
        await self.state.set_setting(LAST_OK_KEY, now_iso)
        await self.state.set_setting(LAST_ERROR_KEY, "")

        # Cursors only advance after the control plane accepted the batch, so a
        # failed sync re-sends rather than skipping rows.
        if prospects:
            await self.state.set_setting(CURSOR_PROSPECTS, _max_updated_at(prospects))
        if emails:
            await self.state.set_setting(CURSOR_OUTBOX, _max_updated_at(emails))

        commands = (response.get("data") or {}).get("commands") or []
        applied = []
        for command in commands:
            applied.append(await self.applier.apply(command, now_iso))
        # Acked on the next cycle: acking in the same request we received them
        # in is impossible, and holding them locally means a crash between the
        # two re-applies an idempotent command rather than losing it.
        self._pending_acks = applied

        return {
            "prospects": len(prospects),
            "emails": len(emails),
            "companies": len(companies),
            "commands": len(applied),
        }

    async def _rows_since(self, table: str, cursor: str) -> list[dict]:
        rows = await self.state.get_rows_updated_since(table, cursor, MAX_BATCH)
        return list(rows or [])

    async def _companies_for(self, prospects: list[dict]) -> list[dict]:
        """Only the companies the pushed prospects actually reference.

        Pushing the whole companies table every cycle would be wasteful, and
        pushing none would leave prospects unattached on the other side.
        """
        wanted = {row.get("company_id") for row in prospects if row.get("company_id")}
        if not wanted:
            return []
        rows = await self.state.get_companies_by_ids(sorted(wanted))
        return list(rows or [])

    # ── the loop ───────────────────────────────────────────────────────────

    async def loop(self, stop_event=None):
        """Runs until stopped. Never raises: a bridge that dies on a bad
        network would leave the operator's approvals stranded with no sign of
        why."""
        import asyncio

        while True:
            if stop_event is not None and stop_event.is_set():
                return
            delay = self.sync_interval
            try:
                await self.sync_once()
            except OpenReplyError as exc:
                delay = min(
                    MAX_BACKOFF_SECONDS,
                    self.sync_interval * (2 ** min(self._consecutive_failures, 6)),
                )
                # Logged once per transition, not once per failure, so an
                # overnight outage does not produce 480 identical lines.
                if self._consecutive_failures == 1:
                    self.log.warning("openreply: sync failing, backing off: %s", exc)
            except Exception as exc:  # a bug here must not take Harvey down
                self.log.exception("openreply: unexpected bridge error: %s", exc)

            if stop_event is not None:
                try:
                    await asyncio.wait_for(stop_event.wait(), timeout=delay)
                    return
                except asyncio.TimeoutError:
                    continue
            else:
                await asyncio.sleep(delay)


def _utcnow_iso() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _max_updated_at(rows: list[dict]) -> str:
    """The cursor for the next cycle.

    Normalised to the space-separated spelling, which is what
    StateManager.get_rows_updated_since compares against. See the note there:
    mixing the two spellings in a cursor makes rows disappear.
    """
    return max(str(row.get("updated_at") or "").replace("T", " ") for row in rows)


def _harvey_version() -> str:
    try:
        import harvey

        return getattr(harvey, "__version__", "") or ""
    except Exception:
        return ""
