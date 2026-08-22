import pytest
from rostam.rostam import _parse_target
from rostam import TransportError


def test_http_scheme():
    assert _parse_target("http://h:8080")[0] == "http"
    assert _parse_target("https://h")[0] == "http"


def test_tcp_scheme_and_bare():
    assert _parse_target("tcp://h:7000")[:3] == ("tcp", "h", 7000)
    assert _parse_target("h:7000")[:3] == ("tcp", "h", 7000)   # bare = native


def test_bare_without_port_errors():
    with pytest.raises(TransportError):
        _parse_target("justahost")
