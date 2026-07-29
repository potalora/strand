"""Process-wide non-loopback network denial for the local-only E2E backend."""

from __future__ import annotations

import socket
from ipaddress import ip_address
from typing import Any


class E2ELocalEgressBlocked(RuntimeError):
    """Raised before an E2E backend socket can contact a non-loopback host."""


def _is_loopback_host(host: object) -> bool:
    if not isinstance(host, str):
        return False
    if host.lower() == "localhost":
        return True
    try:
        return ip_address(host).is_loopback
    except ValueError:
        return False


def assert_loopback_destination(address: object) -> None:
    """Allow Unix sockets and IP loopback only; reject external destinations."""

    if isinstance(address, (str, bytes)):
        return
    if not isinstance(address, tuple) or not address:
        raise E2ELocalEgressBlocked(
            f"Local-only E2E blocked unrecognized socket destination: {address!r}"
        )
    host = address[0]
    if not _is_loopback_host(host):
        raise E2ELocalEgressBlocked(
            f"Local-only E2E blocked non-loopback socket destination: {host!r}"
        )


def install_loopback_socket_guard(*, monkeypatch: Any | None = None) -> None:
    """Patch this process so client sockets can reach loopback endpoints only."""

    original_socket = socket.socket
    original_getaddrinfo = socket.getaddrinfo

    class LoopbackOnlySocket(original_socket):
        def connect(self, address: object) -> None:
            assert_loopback_destination(address)
            return super().connect(address)

        def connect_ex(self, address: object) -> int:
            assert_loopback_destination(address)
            return super().connect_ex(address)

        def sendto(self, data: bytes, *args: object) -> int:
            if args:
                assert_loopback_destination(args[-1])
            return super().sendto(data, *args)

        def sendmsg(
            self,
            buffers: object,
            ancdata: object = (),
            flags: int = 0,
            address: object | None = None,
        ) -> int:
            if address is None:
                return super().sendmsg(buffers, ancdata, flags)
            assert_loopback_destination(address)
            return super().sendmsg(buffers, ancdata, flags, address)

    def loopback_getaddrinfo(
        host: str | bytes | None,
        port: str | int | None,
        *args: object,
        **kwargs: object,
    ) -> list[tuple[Any, ...]]:
        if host is not None:
            decoded_host = host.decode() if isinstance(host, bytes) else host
            if not _is_loopback_host(decoded_host):
                raise E2ELocalEgressBlocked(
                    "Local-only E2E blocked DNS resolution for non-loopback "
                    f"host: {decoded_host!r}"
                )
        return original_getaddrinfo(host, port, *args, **kwargs)

    if monkeypatch is not None:
        monkeypatch.setattr(socket, "socket", LoopbackOnlySocket)
        monkeypatch.setattr(socket, "getaddrinfo", loopback_getaddrinfo)
        return
    socket.socket = LoopbackOnlySocket
    socket.getaddrinfo = loopback_getaddrinfo
