"""The helpers of the local load tests (the ones with a bug in them).

WHAT THIS IS
    The load scripts read the container's memory from docker stats, which
    prints a number with a unit. GiB is 1024^3 bytes and MiB is 1024^2 — the
    first version of the script multiplied both by 1024^2, so a GiB reading
    was off by 1024. This pins the conversion.

HOW TO RUN
        py -m pytest tests/test_local_load_common.py -q
"""
from eval.local_load.common import parse_memory


def test_gib_is_1024_cubed():
    assert parse_memory("1.5GiB") == int(1.5 * 1024 ** 3)


def test_mib_is_1024_squared():
    assert parse_memory("62.1MiB") == int(62.1 * 1024 ** 2)


def test_a_reading_without_a_unit_is_zero():
    assert parse_memory("--") == 0
    assert parse_memory("") == 0


def test_only_the_usage_is_parsed_not_the_limit():
    # docker stats prints "used / limit"; the limit must never be read as a usage
    assert parse_memory("0B / 4GiB") == 0
    assert parse_memory("55.5MiB / 4GiB") == int(55.5 * 1024 ** 2)


def test_the_first_unit_in_the_text_wins():
    assert parse_memory("1.5GiB 62.1MiB") == int(1.5 * 1024 ** 3)