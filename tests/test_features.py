from src.features import canonical_host, extract_features, extract_host_features


def test_basic_https_url():
    f = extract_features("https://www.example.com/login?user=1&x=2")
    assert f["is_https"] == 1
    assert f["is_ip"] == 0
    assert f["num_qmark"] == 1
    assert f["num_amp"] == 1
    assert f["tld"] == "com"
    assert f["num_suspicious_words"] >= 1


def test_ip_host():
    assert extract_features("http://192.168.1.1/a")["is_ip"] == 1


def test_missing_scheme():
    f = extract_features("example.com/path")
    assert f["is_https"] == 0
    assert f["domain_length"] == len("example.com")


def test_empty_string_does_not_crash():
    f = extract_features("")
    assert f["url_length"] == 0
    assert f["entropy"] == 0.0


def test_canonical_host_strips_scheme_www_path():
    assert canonical_host("https://www.Example.com/a/b?x=1") == "example.com"
    assert canonical_host("example.com/path") == "example.com"
    assert canonical_host("http://sub.example.co.uk:8080/x") == "sub.example.co.uk"


def test_host_features_ignore_scheme_www_and_path():
    a = extract_host_features("https://www.example.com/login?x=1")
    b = extract_host_features("http://example.com")
    assert a == b


def test_host_features_empty():
    assert extract_host_features("")["host_length"] == 0