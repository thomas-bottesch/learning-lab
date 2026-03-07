import html
import os
import re
from urllib.parse import urljoin

import kfp
import requests


def _extract_login_form(html_text: str) -> tuple[str, dict[str, str]]:
    form_match = re.search(
        r"<form[^>]*action=\"([^\"]+)\"[^>]*>", html_text, re.IGNORECASE
    )
    if not form_match:
        raise RuntimeError("Could not find Dex login form in response HTML.")

    action = html.unescape(form_match.group(1))
    hidden_inputs = {
        html.unescape(key): html.unescape(value)
        for key, value in re.findall(
            r"<input[^>]*type=\"hidden\"[^>]*name=\"([^\"]+)\"[^>]*value=\"([^\"]*)\"",
            html_text,
            re.IGNORECASE,
        )
    }
    return action, hidden_inputs


def _extract_form(
    html_text: str, action_pattern: str
) -> tuple[str, dict[str, str]] | None:
    form_match = re.search(
        rf"<form[^>]*action=\"([^\"]*{action_pattern}[^\"]*)\"[^>]*>",
        html_text,
        re.IGNORECASE,
    )
    if not form_match:
        return None

    action = html.unescape(form_match.group(1))
    hidden_inputs = {
        html.unescape(key): html.unescape(value)
        for key, value in re.findall(
            r"<input[^>]*type=\"hidden\"[^>]*name=\"([^\"]+)\"[^>]*value=\"([^\"]*)\"",
            html_text,
            re.IGNORECASE,
        )
    }
    return action, hidden_inputs


def _ensure_pipeline_host(host: str) -> str:
    cleaned = host.rstrip("/")
    if cleaned.endswith("/pipeline"):
        return cleaned
    return f"{cleaned}/pipeline"


def _host_without_pipeline(host: str) -> str:
    cleaned = host.rstrip("/")
    if cleaned.endswith("/pipeline"):
        return cleaned[: -len("/pipeline")]
    return cleaned


def get_authservice_cookie(
    host: str, username: str, password: str, timeout: int = 30
) -> str:
    session = requests.Session()
    ingress_host = _host_without_pipeline(host)

    resp = session.get(
        f"{ingress_host}/pipeline", allow_redirects=True, timeout=timeout
    )

    oauth2_start_form = _extract_form(resp.text, r"/oauth2/start")
    if oauth2_start_form is not None:
        start_action, start_hidden = oauth2_start_form
        start_url = urljoin(resp.url, start_action)
        start_resp = session.post(
            start_url,
            data=start_hidden,
            allow_redirects=True,
            timeout=timeout,
        )
        start_resp.raise_for_status()
        login_page = start_resp
    else:
        resp.raise_for_status()
        login_page = resp

    action, hidden = _extract_login_form(login_page.text)
    login_url = urljoin(login_page.url, action)

    payload = {
        **hidden,
        "login": username,
        "password": password,
    }

    login_resp = session.post(
        login_url, data=payload, allow_redirects=True, timeout=timeout
    )
    login_resp.raise_for_status()

    cookie_name_candidates = [
        "authservice_session",
        "oauth2_proxy",
        "oauth2_proxy_kubeflow",
        "__Host-authservice_session",
    ]

    cookie_name = None
    cookie_value = None
    for candidate in cookie_name_candidates:
        value = session.cookies.get(candidate)
        if value:
            cookie_name = candidate
            cookie_value = value
            break

    if not cookie_name or not cookie_value:
        raise RuntimeError(
            "Dex login completed but no supported auth cookie was found "
            "(authservice_session/oauth2_proxy). Check host URL, credentials, and Dex configuration."
        )
    return f"{cookie_name}={cookie_value}"


def get_kfp_client(
    host: str | None = None,
    namespace: str | None = None,
    cookie: str | None = None,
    token: str | None = None,
    username: str | None = None,
    password: str | None = None,
) -> kfp.Client:
    host_value = _ensure_pipeline_host(host or os.getenv("KUBEFLOW_HOST"))
    namespace_value = namespace or os.getenv("KUBEFLOW_NAMESPACE")
    cookie_value = cookie or os.getenv("KUBEFLOW_COOKIE")
    token_value = token or os.getenv("KUBEFLOW_TOKEN")
    username_value = username or os.getenv("KUBEFLOW_USERNAME")
    password_value = password or os.getenv("KUBEFLOW_PASSWORD")

    if not cookie_value and not token_value and username_value and password_value:
        cookie_value = get_authservice_cookie(
            host_value, username_value, password_value
        )
    print(f"Using KUBEFLOW_HOST: {host_value}")
    print(f"Using KUBEFLOW_NAMESPACE: {namespace_value}")
    client_kwargs: dict[str, str] = {
        "host": host_value,
        "namespace": namespace_value,
    }
    if cookie_value:
        client_kwargs["cookies"] = cookie_value
    if token_value:
        client_kwargs["existing_token"] = token_value

    return kfp.Client(**client_kwargs)


if __name__ == "__main__":
    print("Testing KFP client authentication...")
    try:
        client = get_kfp_client()
        print("Successfully created KFP client with provided credentials.")
    except Exception as err:
        message = str(err)
        if (
            "401" in message
            or "Unauthorized" in message
            or "User identity is empty" in message
        ):
            print("Authentication failed for Kubeflow Pipelines.")
        print(message)
