import json
from typing import Any, Optional
import urllib.error
import urllib.parse
import urllib.request

from tools.base import Tool


class ApiTool(Tool):
    def __init__(
        self,
        name: str,
        description: str,
        url: str,
        method: str = "GET",
        parameters: Optional[list[dict[str, Any]]] = None,
        headers: Optional[dict[str, str]] = None,
        response_template: Optional[str] = None,
        timeout: float = 5.0,
    ):
        super().__init__(name, description, parameters, response_template)
        self.url = url.strip()
        self.method = method.upper()
        self.headers = headers or {"User-Agent": "AI-Institute/1.0"}
        self.timeout = timeout

        parsed = urllib.parse.urlparse(self.url)
        if parsed.scheme not in ("http", "https"):
            raise ValueError(f"Invalid URL scheme '{parsed.scheme}'. Only http and https are allowed.")

    def execute(self, arguments: dict[str, Any]) -> Any:
        url = self.url
        req_headers = dict(self.headers)
        data = None

        if self.method == "GET":
            # Append arguments as query params
            if arguments:
                query_string = urllib.parse.urlencode(arguments)
                join_char = "&" if "?" in url else "?"
                url = f"{url}{join_char}{query_string}"
        elif self.method == "POST":
            req_headers["Content-Type"] = "application/json"
            data = json.dumps(arguments).encode("utf-8")

        req = urllib.request.Request(url, data=data, headers=req_headers, method=self.method)

        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as response:
                content_type = response.headers.get("Content-Type", "")
                raw_bytes = response.read()
                raw_text = raw_bytes.decode("utf-8", errors="replace")

                if "application/json" in content_type or raw_text.strip().startswith(("{", "[")):
                    try:
                        return json.loads(raw_text)
                    except json.JSONDecodeError:
                        return {"raw": raw_text}
                return {"data": raw_text}
        except urllib.error.HTTPError as e:
            return {"error": f"HTTP {e.code}: {e.reason}"}
        except urllib.error.URLError as e:
            return {"error": f"Connection error: {e.reason}"}
        except Exception as e:
            return {"error": str(e)}
