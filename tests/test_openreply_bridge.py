"""Bridge signing tests. Stdlib only, so they run without network or extras."""

import pytest

from harvey.integrations.openreply import (
    OpenReplyClient,
    OpenReplyError,
    build_headers,
    canonical_body,
    sign_payload,
)

# The same vector asserted in openreply/__tests__/harvey-signature.test.ts.
# Both sides must agree or the bridge 401s in production.
GOLDEN_BODY = '{"alpha":2,"kind":"sync","prospects":[{"email":"a@b.co","id":"p1"}],"zeta":1}'
GOLDEN_TIMESTAMP = "1760000000"
GOLDEN_SECRET = "shared-secret"
GOLDEN_SIGNATURE = "sha256=1d44b02d29d2b1f440b6561f23669c6ae0eebf75331336bb665a348713786406"


def test_canonical_body_is_stable_and_compact():
    payload = {"zeta": 1, "kind": "sync", "prospects": [{"id": "p1", "email": "a@b.co"}], "alpha": 2}
    assert canonical_body(payload) == GOLDEN_BODY
    # Key order in the source dict must not change the bytes on the wire.
    assert canonical_body(dict(reversed(list(payload.items())))) == GOLDEN_BODY


def test_golden_signature_matches_the_typescript_verifier():
    assert sign_payload(GOLDEN_BODY, GOLDEN_TIMESTAMP, GOLDEN_SECRET) == GOLDEN_SIGNATURE


def test_signature_covers_the_timestamp():
    assert sign_payload(GOLDEN_BODY, "1760000001", GOLDEN_SECRET) != GOLDEN_SIGNATURE


def test_signature_covers_the_body():
    assert sign_payload(GOLDEN_BODY + " ", GOLDEN_TIMESTAMP, GOLDEN_SECRET) != GOLDEN_SIGNATURE


def test_build_headers_sends_what_it_signed():
    headers = build_headers(GOLDEN_BODY, "agent-1", GOLDEN_SECRET, GOLDEN_TIMESTAMP)
    assert headers["X-Harvey-Agent"] == "agent-1"
    assert headers["X-Harvey-Timestamp"] == GOLDEN_TIMESTAMP
    assert headers["X-Harvey-Signature"] == GOLDEN_SIGNATURE
    assert headers["Content-Type"] == "application/json"


def test_build_headers_defaults_the_timestamp_to_now():
    headers = build_headers(GOLDEN_BODY, "agent-1", GOLDEN_SECRET)
    assert headers["X-Harvey-Timestamp"].isdigit()


def test_is_configured_requires_all_three_settings():
    assert OpenReplyClient("https://x.test", "agent-1", "s").is_configured()
    assert not OpenReplyClient("", "agent-1", "s").is_configured()
    assert not OpenReplyClient("https://x.test", "", "s").is_configured()
    assert not OpenReplyClient("https://x.test", "node-1", "").is_configured()
    # Whitespace-only values are not configuration.
    assert not OpenReplyClient("https://x.test", "  ", "s").is_configured()


def test_base_url_trailing_slash_is_dropped():
    assert OpenReplyClient("https://x.test/", "n", "s").base_url == "https://x.test"


@pytest.mark.asyncio
async def test_post_refuses_when_unconfigured():
    with pytest.raises(OpenReplyError, match="not configured"):
        await OpenReplyClient("", "", "").post("/api/harvey/agent/sync", {})


# ─────────────────────────────────────────────────────────────────────────────
# Body rewriting
# ─────────────────────────────────────────────────────────────────────────────

from harvey.integrations.openreply import (  # noqa: E402
    CommandApplier,
    DEFAULT_LINK_CTA,
    OpenReplyBridge,
    body_has_tracked_link,
    command_is_expired,
    company_payload,
    email_payload,
    prospect_payload,
    rewrite_body,
    should_mint_link,
    _as_wire_timestamp,
    _max_updated_at,
)

BASE_URL = "https://openreply.example"
TRACKED = f"{BASE_URL}/r/hv_9fQ2xs"


def test_append_adds_one_line_and_keeps_the_body():
    out = rewrite_body("Saw you stock Jordans.", TRACKED, mode="append")
    assert out.startswith("Saw you stock Jordans.")
    assert out.count(TRACKED) == 1
    assert DEFAULT_LINK_CTA in out


def test_replace_swaps_the_writers_url_so_there_is_still_one_link():
    body = "Book a slot: https://cal.com/vantik/intro"
    out = rewrite_body(
        body, TRACKED, mode="replace", placeholder_url="https://cal.com/vantik/intro"
    )
    assert out == f"Book a slot: {TRACKED}"
    assert "cal.com" not in out


def test_replace_falls_back_to_append_when_the_url_is_absent():
    out = rewrite_body("No link in here.", TRACKED, mode="replace",
                       placeholder_url="https://cal.com/vantik/intro")
    assert TRACKED in out
    assert out.startswith("No link in here.")


def test_none_mode_never_touches_the_body():
    assert rewrite_body("As written.", TRACKED, mode="none") == "As written."


def test_an_unknown_mode_is_a_programming_error_not_a_silent_default():
    with pytest.raises(ValueError, match="link mode"):
        rewrite_body("x", TRACKED, mode="sideways")


def test_body_with_no_tracked_url_is_left_alone_when_there_is_nothing_to_add():
    assert rewrite_body("Body", "", mode="append") == "Body"


def test_tracked_link_detection_ignores_other_hosts():
    assert body_has_tracked_link(f"see {TRACKED}", BASE_URL)
    assert not body_has_tracked_link("see https://elsewhere.test/r/abc", BASE_URL)
    assert not body_has_tracked_link("no links", BASE_URL)
    # With no base url configured there is nothing to recognise.
    assert not body_has_tracked_link(f"see {TRACKED}", "")


def test_detection_tolerates_a_trailing_slash_in_the_base_url():
    assert body_has_tracked_link(f"see {TRACKED}", BASE_URL + "/")


# ─────────────────────────────────────────────────────────────────────────────
# Which emails get a link
# ─────────────────────────────────────────────────────────────────────────────


def _item(**kwargs):
    return {"id": "o1", "status": "pending_review", "kind": "sequence", "step": 2,
            "body": "hello", **kwargs}


def test_step_one_never_gets_a_link():
    # Harvey's own writer prompt forbids it, and a first touch that asks a
    # question beats one that asks for a click.
    assert not should_mint_link(_item(step=1), base_url=BASE_URL)


def test_step_two_onwards_does():
    assert should_mint_link(_item(step=2), base_url=BASE_URL)
    assert should_mint_link(_item(step=3), base_url=BASE_URL)


def test_a_body_that_already_has_a_link_is_not_minted_again():
    assert not should_mint_link(
        _item(body=f"hello {TRACKED}"), base_url=BASE_URL
    )


def test_only_pending_or_approved_items_are_minted():
    for status in ("sent", "failed", "cancelled"):
        assert not should_mint_link(_item(status=status), base_url=BASE_URL)
    assert should_mint_link(_item(status="approved"), base_url=BASE_URL)


def test_replies_are_opt_in():
    assert not should_mint_link(_item(kind="reply"), base_url=BASE_URL)
    assert should_mint_link(_item(kind="reply"), base_url=BASE_URL, link_in_replies=True)


def test_an_unrecognised_kind_is_skipped():
    assert not should_mint_link(_item(kind="newsletter"), base_url=BASE_URL)


def test_a_malformed_step_is_skipped_rather_than_crashing_the_cycle():
    assert not should_mint_link(_item(step="soon"), base_url=BASE_URL)


# ─────────────────────────────────────────────────────────────────────────────
# Payloads
# ─────────────────────────────────────────────────────────────────────────────


def test_prospect_payload_drops_an_email_status_the_wire_will_not_accept():
    assert prospect_payload({"id": "p1", "email_status": "verified"})["emailStatus"] == "verified"
    assert prospect_payload({"id": "p1", "email_status": ""})["emailStatus"] is None
    assert prospect_payload({"id": "p1", "email_status": "weird"})["emailStatus"] is None


def test_prospect_payload_maps_snake_case_to_the_wire_names():
    payload = prospect_payload({
        "id": "p1", "company_id": "c1", "first_name": "Amina", "last_name": "Okello",
        "email": "amina@kampalakicks.ug", "title": "Owner", "status": "contacted",
        "score": 78, "personalization_notes": "Stocks Jordan 1s",
    })
    assert payload == {
        "harveyId": "p1", "companyHarveyId": "c1", "firstName": "Amina",
        "lastName": "Okello", "email": "amina@kampalakicks.ug", "emailStatus": None,
        "title": "Owner", "status": "contacted", "score": 78,
        "personalizationNotes": "Stocks Jordan 1s",
    }


def test_company_payload_sends_null_not_empty_string():
    payload = company_payload({"id": "c1", "name": "Kampala Kicks", "domain": ""})
    assert payload["domain"] is None
    assert payload["name"] == "Kampala Kicks"


def test_email_payload_defaults_a_missing_step_to_one():
    assert email_payload({"id": "o1"})["step"] == 1


def test_naive_sqlite_timestamps_are_sent_as_utc():
    # Without the offset, a send time would be read as local to whatever
    # region the control plane's server sits in.
    assert _as_wire_timestamp("2026-10-15 09:30:00") == "2026-10-15T09:30:00+00:00"
    assert _as_wire_timestamp("2026-10-15T09:30:00+00:00") == "2026-10-15T09:30:00+00:00"
    assert _as_wire_timestamp("2026-10-15T09:30:00Z") == "2026-10-15T09:30:00Z"
    assert _as_wire_timestamp("") is None
    assert _as_wire_timestamp(None) is None


def test_cursor_normalises_the_two_timestamp_spellings():
    # "T" sorts above " ", so an un-normalised cursor taken from the ISO row
    # would skip the genuinely newer space-separated one for good.
    rows = [
        {"updated_at": "2026-10-15T00:00:00"},
        {"updated_at": "2026-10-15 23:59:59"},
    ]
    assert _max_updated_at(rows) == "2026-10-15 23:59:59"


# ─────────────────────────────────────────────────────────────────────────────
# Applying the operator's decisions
# ─────────────────────────────────────────────────────────────────────────────


class FakeState:
    """Just enough of StateManager to exercise the applier, with no database."""

    def __init__(self, items=None):
        self.items = items or {}
        self.settings = {}
        self.updates = []
        self.approve_all_calls = 0

    async def get_outbox_item(self, item_id):
        return self.items.get(item_id)

    async def approve_outbox(self, item_id=None):
        if item_id is None:
            self.approve_all_calls += 1
            moved = [i for i in self.items.values() if i["status"] == "pending_review"]
            for item in moved:
                item["status"] = "approved"
            return len(moved)
        item = self.items.get(item_id)
        if item and item["status"] == "pending_review":
            item["status"] = "approved"
            return 1
        return 0

    async def update_outbox_item(self, item_id, **fields):
        self.updates.append((item_id, fields))
        self.items.setdefault(item_id, {}).update(fields)

    async def get_setting(self, key, default=""):
        return self.settings.get(key, default)

    async def set_setting(self, key, value):
        self.settings[key] = value


def _command(kind="APPROVE_EMAIL", target="o1", **kwargs):
    return {"id": "cmd_1", "kind": kind, "targetHarveyId": target, **kwargs}


NOW = "2026-10-15T09:00:00+00:00"


@pytest.mark.asyncio
async def test_approve_moves_a_waiting_email():
    state = FakeState({"o1": {"id": "o1", "status": "pending_review"}})
    result = await CommandApplier(state).apply(_command(), NOW)
    assert result == {"commandId": "cmd_1", "result": "approved"}
    assert state.items["o1"]["status"] == "approved"


@pytest.mark.asyncio
async def test_approving_twice_is_safe_and_says_so():
    state = FakeState({"o1": {"id": "o1", "status": "pending_review"}})
    applier = CommandApplier(state)
    await applier.apply(_command(), NOW)
    second = await applier.apply(_command(), NOW)
    # Delivered at least once means this will happen; it must not double-send.
    assert second["result"] == "already_approved"


@pytest.mark.asyncio
async def test_approving_something_already_sent_reports_it_rather_than_failing():
    state = FakeState({"o1": {"id": "o1", "status": "sent"}})
    result = await CommandApplier(state).apply(_command(), NOW)
    assert result["result"] == "already_sent"


@pytest.mark.asyncio
async def test_reject_cancels_with_the_reason_recorded():
    state = FakeState({"o1": {"id": "o1", "status": "pending_review"}})
    result = await CommandApplier(state).apply(
        _command(kind="REJECT_EMAIL", payload={"reason": "wrong contact"}), NOW
    )
    assert result["result"] == "rejected"
    assert state.updates == [("o1", {"status": "cancelled", "error": "wrong contact"})]


@pytest.mark.asyncio
async def test_reject_without_a_reason_still_records_where_it_came_from():
    state = FakeState({"o1": {"id": "o1", "status": "pending_review"}})
    await CommandApplier(state).apply(_command(kind="REJECT_EMAIL"), NOW)
    assert "OpenReply" in state.updates[0][1]["error"]


@pytest.mark.asyncio
async def test_edit_rewrites_a_waiting_email():
    state = FakeState({"o1": {"id": "o1", "status": "pending_review"}})
    result = await CommandApplier(state).apply(
        _command(kind="EDIT_EMAIL", payload={"subject": "wholesale list", "body": "Fixed."}),
        NOW,
    )
    assert result["result"] == "edited"
    assert state.updates == [("o1", {"subject": "wholesale list", "body": "Fixed."})]


@pytest.mark.asyncio
async def test_edit_refuses_an_email_already_out_the_door():
    # Rewriting a sent email would make the record lie about what was read.
    state = FakeState({"o1": {"id": "o1", "status": "sent"}})
    result = await CommandApplier(state).apply(
        _command(kind="EDIT_EMAIL", payload={"body": "too late"}), NOW
    )
    assert result["result"] == "already_sent"
    assert state.updates == []


@pytest.mark.asyncio
async def test_an_empty_edit_changes_nothing():
    state = FakeState({"o1": {"id": "o1", "status": "pending_review"}})
    result = await CommandApplier(state).apply(_command(kind="EDIT_EMAIL", payload={}), NOW)
    assert result["result"] == "nothing_to_edit"
    assert state.updates == []


@pytest.mark.asyncio
async def test_pause_uses_harveys_own_kill_switch_with_a_readable_reason():
    state = FakeState()
    await CommandApplier(state).apply(_command(kind="PAUSE_SENDING", target=""), NOW)
    # harvey/agents/sender.py reads this exact key, and the CLI prints the value.
    assert "OpenReply" in state.settings["sending_paused"]

    await CommandApplier(state).apply(_command(kind="RESUME_SENDING", target=""), NOW)
    assert state.settings["sending_paused"] == ""


@pytest.mark.asyncio
async def test_approve_all_reports_how_many_it_moved():
    state = FakeState({
        "o1": {"id": "o1", "status": "pending_review"},
        "o2": {"id": "o2", "status": "pending_review"},
        "o3": {"id": "o3", "status": "sent"},
    })
    result = await CommandApplier(state).apply(_command(kind="APPROVE_ALL", target=""), NOW)
    assert result["result"] == "approved_all:2"


@pytest.mark.asyncio
async def test_a_command_for_an_unknown_email_is_acked_not_retried_forever():
    state = FakeState()
    result = await CommandApplier(state).apply(_command(target="ghost"), NOW)
    assert result["result"] == "not_found"


@pytest.mark.asyncio
async def test_a_command_kind_from_a_newer_control_plane_is_acked_with_a_reason():
    state = FakeState()
    result = await CommandApplier(state).apply(_command(kind="LAUNCH_ROCKET"), NOW)
    assert result["result"] == "unsupported:LAUNCH_ROCKET"


@pytest.mark.asyncio
async def test_an_expired_command_is_not_applied():
    # The laptop was closed for three days. Monday's approval must not fire
    # against a prospect whose situation has moved on.
    state = FakeState({"o1": {"id": "o1", "status": "pending_review"}})
    result = await CommandApplier(state).apply(
        _command(expiresAt="2026-10-14T00:00:00+00:00"), NOW
    )
    assert result["result"] == "expired"
    assert state.items["o1"]["status"] == "pending_review"


@pytest.mark.asyncio
async def test_a_command_expiring_in_the_future_still_applies():
    state = FakeState({"o1": {"id": "o1", "status": "pending_review"}})
    result = await CommandApplier(state).apply(
        _command(expiresAt="2026-10-18T00:00:00+00:00"), NOW
    )
    assert result["result"] == "approved"


@pytest.mark.asyncio
async def test_a_handler_blowing_up_does_not_stall_the_queue_behind_it():
    class Exploding(FakeState):
        async def get_outbox_item(self, item_id):
            raise RuntimeError("disk gone")

    result = await CommandApplier(Exploding()).apply(_command(), NOW)
    assert result["result"].startswith("error:")


def test_expiry_check_treats_a_missing_expiry_as_no_expiry():
    assert not command_is_expired({}, NOW)
    assert not command_is_expired({"expiresAt": None}, NOW)
