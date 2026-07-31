from __future__ import annotations

import socket

import pytest

from tests.e2e_local_network_guard import (
    E2ELocalEgressBlocked,
    assert_loopback_destination,
    install_loopback_socket_guard,
)


@pytest.mark.parametrize(
    "address",
    [
        ("localhost", 8000),
        ("127.0.0.1", 5432),
        ("127.12.34.56", 6379),
        ("::1", 8000),
    ],
)
def test_loopback_destinations_are_allowed(address: tuple[str, int]) -> None:
    assert_loopback_destination(address)


@pytest.mark.parametrize(
    "address",
    [
        ("db.example.com", 5432),
        ("192.0.2.10", 443),
        ("2001:db8::10", 443),
    ],
)
def test_non_loopback_destinations_are_rejected(address: tuple[str, int]) -> None:
    with pytest.raises(E2ELocalEgressBlocked, match="non-loopback"):
        assert_loopback_destination(address)


def test_installed_socket_guard_blocks_before_connect(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_loopback_socket_guard(monkeypatch=monkeypatch)

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client:
        with pytest.raises(E2ELocalEgressBlocked, match="192.0.2.10"):
            client.connect(("192.0.2.10", 443))


def test_installed_dns_guard_rejects_external_hostnames(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_loopback_socket_guard(monkeypatch=monkeypatch)

    with pytest.raises(E2ELocalEgressBlocked, match="example.com"):
        socket.getaddrinfo("example.com", 443)


def test_installed_guard_preserves_addressless_local_sendmsg(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_loopback_socket_guard(monkeypatch=monkeypatch)

    sender, receiver = socket.socketpair()
    with sender, receiver:
        sender.sendmsg([b"local"])
        assert receiver.recv(5) == b"local"
