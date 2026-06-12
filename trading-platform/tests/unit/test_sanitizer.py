"""Unit tests for ``security.sanitizer`` — log redaction patterns."""

from __future__ import annotations

import sys
from pathlib import Path

_repo = Path(__file__).resolve().parents[2]
for p in [_repo, _repo / "packages" / "security" / "src", _repo / "packages"]:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import pytest

from security.sanitizer import LogSanitizer, _REDACTED, sanitize, sanitize_log


class TestApiKeyPatterns:
    def test_api_key_equals_value(self) -> None:
        assert sanitize("api_key=supersecret") == _REDACTED

    def test_apikey_colon_value(self) -> None:
        assert sanitize("apikey:abc123token") == _REDACTED

    def test_api_key_space_equals_spaced_value(self) -> None:
        assert sanitize("API-KEY = myvalue") == _REDACTED

    def test_api_secret_equals(self) -> None:
        assert sanitize("api_secret=hidden") == _REDACTED

    def test_secret_key_colon(self) -> None:
        assert sanitize("secret_key:xyz") == _REDACTED

    def test_api_key_case_insensitive_label(self) -> None:
        out = sanitize("Api_Key=val")
        assert _REDACTED in out
        assert "val" not in out


class TestPasswordPatterns:
    def test_password_equals(self) -> None:
        assert sanitize("password=hunter2") == _REDACTED

    def test_passwd_colon(self) -> None:
        assert sanitize("passwd:secret") == _REDACTED

    def test_pwd_equals(self) -> None:
        assert sanitize("pwd=letmein") == _REDACTED

    def test_password_case_insensitive(self) -> None:
        assert sanitize("PASSWORD=top") == _REDACTED


class TestTokenPatterns:
    def test_token_space_value(self) -> None:
        assert sanitize("token abc123") == _REDACTED

    def test_bearer_space_value(self) -> None:
        assert sanitize("bearer abc123") == _REDACTED

    def test_bearer_uppercase_label(self) -> None:
        assert sanitize("BEARER xyz") == _REDACTED


class TestAuthorizationHeader:
    def test_authorization_bearer(self) -> None:
        assert sanitize("authorization: Bearer xyz") == _REDACTED


class TestOpenAiKeys:
    def test_sk_prefix_minimum_length(self) -> None:
        tail = "a" * 20
        out = sanitize(f"sk-{tail}")
        assert out == _REDACTED
        assert tail not in out

    def test_sk_key_middle_of_sentence(self) -> None:
        sk = "sk-" + "b" * 22
        out = sanitize(f"using {sk} here")
        assert _REDACTED in out
        assert "bbbb" not in out


class TestAwsKeys:
    def test_akia_access_key(self) -> None:
        key = "AKIA" + "0" * 16
        out = sanitize(f"creds {key} end")
        assert _REDACTED in out
        assert "AKIA" not in out

    def test_asia_sts_key(self) -> None:
        key = "ASIA" + "1" * 16
        out = sanitize(f"{key}")
        assert out == _REDACTED


class TestAccountNumbers:
    def test_account_space_digits(self) -> None:
        assert sanitize("account 1234567") == _REDACTED

    def test_acct_hash_digits(self) -> None:
        assert sanitize("acct#123456") == _REDACTED

    def test_acct_colon_space_digits(self) -> None:
        assert sanitize("acct: 1234567") == _REDACTED


class TestPhoneLikeNumbers:
    def test_phone_dashes(self) -> None:
        assert sanitize("call 123-456-7890") == f"call {_REDACTED}"

    def test_phone_no_separators(self) -> None:
        assert sanitize("1234567890") == _REDACTED


class TestLongOpaqueTokens:
    def test_thirty_two_alphanumeric(self) -> None:
        token = "a" * 32
        assert sanitize(f"x {token} y") == f"x {_REDACTED} y"


class TestCardLikeNumbers:
    def test_card_dashed(self) -> None:
        assert sanitize("card 1234-5678-9012-3456") == f"card {_REDACTED}"


class TestEmailAddresses:
    def test_simple_email(self) -> None:
        assert sanitize("user@example.com") == _REDACTED


class TestMixedAndPositioning:
    def test_mixed_multiple_sensitive_items(self) -> None:
        text = "api_key=1 password=2 token abc user@x.co 123-456-7890"
        out = sanitize(text)
        assert out.count(_REDACTED) >= 4
        assert "api_key=1" not in out
        assert "password=2" not in out
        assert "user@x.co" not in out

    def test_sensitive_at_start(self) -> None:
        assert sanitize("password=secret rest") == f"{_REDACTED} rest"

    def test_sensitive_at_end(self) -> None:
        assert sanitize("prefix password=secret") == f"prefix {_REDACTED}"

    def test_sensitive_in_middle(self) -> None:
        assert sanitize("a password=secret b") == f"a {_REDACTED} b"

    def test_multiple_same_pattern(self) -> None:
        out = sanitize("password=a password=b")
        assert out.count(_REDACTED) == 2
        assert "password=a" not in out


class TestUnchangedAndEdgeCases:
    def test_plain_text_unchanged(self) -> None:
        s = "hello world nothing sensitive here"
        assert sanitize(s) == s

    def test_empty_string(self) -> None:
        assert sanitize("") == ""

    def test_short_alphanumeric_unchanged(self) -> None:
        s = "abcdefghijklmnop"  # 16 chars
        assert sanitize(s) == s


class TestLogSanitizerExtraPatterns:
    def test_extra_pattern_redacts(self) -> None:
        custom = LogSanitizer([r"PROJECT[_-]?ID\s*=\s*\S+"])
        out = custom.sanitize("PROJECT_ID=99")
        assert out == _REDACTED

    def test_extra_pattern_preserves_builtin(self) -> None:
        custom = LogSanitizer([r"MYSECRET\s+\S+"])
        out = custom.sanitize("MYSECRET foo password=bar")
        assert _REDACTED in out
        assert "password=bar" not in out

    def test_builtin_still_applies_with_extra(self) -> None:
        s = LogSanitizer([r"foo"])
        assert s.sanitize("password=x") == _REDACTED


class TestModuleFunctions:
    def test_sanitize_matches_default_instance(self) -> None:
        text = "token abc"
        assert sanitize(text) == LogSanitizer().sanitize(text)

    def test_sanitize_log_delegates_to_sanitize(self) -> None:
        text = "bearer xyz"
        assert sanitize_log(text) == sanitize(text)


class TestCaseInsensitivity:
    @pytest.mark.parametrize(
        "line",
        [
            "Api_Key=v",
            "APIKEY:v",
            "TOKEN t",
            "Authorization: x",
            "ACCOUNT 123456",
        ],
    )
    def test_labels_case_insensitive(self, line: str) -> None:
        assert _REDACTED in sanitize(line)


class TestRedactedConstant:
    def test_redacted_marker_value(self) -> None:
        assert _REDACTED == "[REDACTED]"


@pytest.mark.parametrize(
    "raw,expect_substr_absent",
    [
        ("api_key=hidden", "hidden"),
        ("secret_key:x", "x"),
        ("authorization: Bearer z", "z"),
        ("sk-" + "c" * 20, "cccc"),
    ],
)
def test_secrets_removed_from_output(raw: str, expect_substr_absent: str) -> None:
    out = sanitize(raw)
    assert expect_substr_absent not in out


def test_phone_with_spaces() -> None:
    assert sanitize("123 456 7890") == _REDACTED


def test_card_with_spaces() -> None:
    assert sanitize("1234 5678 9012 3456") == _REDACTED


def test_email_with_plus() -> None:
    assert sanitize("a+b@c.co.uk") == _REDACTED


def test_acct_label_lowercase() -> None:
    assert sanitize("acct 888888") == _REDACTED


def test_api_secret_hyphen_form() -> None:
    assert sanitize("api-secret=zzz") == _REDACTED


def test_passwd_spacing_around_equals() -> None:
    assert sanitize("passwd = x") == _REDACTED


def test_token_with_underscore_and_dot() -> None:
    out = sanitize("token ab_c.d-1")
    assert out == _REDACTED


def test_bearer_at_start() -> None:
    assert sanitize("bearer startval trailing") == f"{_REDACTED} trailing"


def test_aws_key_at_string_edges() -> None:
    key = "AKIA" + "A" * 16
    assert sanitize(f"{key}") == _REDACTED
    # Non-word delimiters so ``\b`` matches the AWS key token boundaries.
    assert sanitize(f";{key};") == f";{_REDACTED};"


def test_multiple_phones() -> None:
    out = sanitize("111-222-3333 and 444-555-6666")
    assert out.count(_REDACTED) == 2


def test_long_token_at_edges() -> None:
    t = "z" * 35
    assert sanitize(f"{t}") == _REDACTED
    assert sanitize(f"::{t}::") == f"::{_REDACTED}::"


def test_mixed_case_email_domain() -> None:
    assert sanitize("u@Example.COM") == _REDACTED


def test_sanitize_log_empty() -> None:
    assert sanitize_log("") == ""


def test_log_sanitizer_default_extra_empty_tuple() -> None:
    s = LogSanitizer()
    assert s.sanitize("ok") == "ok"


def test_sequential_redaction_order_independent_assertion() -> None:
    """Overlapping redactions still yield only redaction markers and safe text."""
    out = sanitize("email a@b.co phone 999-888-7777")
    assert "a@b.co" not in out
    assert "999-888-7777" not in out


def test_password_colon_no_space() -> None:
    assert sanitize("password:nospace") == _REDACTED


def test_apikey_equals_compact() -> None:
    assert sanitize("apikey=compact") == _REDACTED

