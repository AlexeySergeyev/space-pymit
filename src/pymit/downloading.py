import urllib.parse
import warnings
from pathlib import Path
from typing import Union

import requests
from urllib3.exceptions import InsecureRequestWarning

from .errors import AsteroidModelError


def _is_http_url(source: str) -> bool:
    parsed = urllib.parse.urlparse(source)
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def _download_lightcurve_url(url: str, output_dir: Union[str, Path]) -> str:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    parsed = urllib.parse.urlparse(url)
    filename = Path(parsed.path).name or "downloaded_lightcurve.txt"
    destination = output_path / filename

    try:
        response = requests.get(url, timeout=30, verify=True)
    except requests.exceptions.SSLError as e:
        if parsed.netloc == "damit.cuni.cz":
            warnings.warn(
                "Verified HTTPS download from damit.cuni.cz failed certificate validation; "
                "retrying with certificate verification disabled for this DAMIT plaintext export.",
                RuntimeWarning,
                stacklevel=2,
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", InsecureRequestWarning)
                response = requests.get(url, timeout=30, verify=False)
        else:
            raise AsteroidModelError(
                "SSL certificate verification failed while downloading the DAMIT lightcurve. "
                "Install or update a certificate bundle, for example with "
                "`python3 -m pip install certifi`, then rerun the script."
            ) from e
    except requests.exceptions.RequestException as e:
        raise AsteroidModelError(f"Failed to download lightcurve from {url}: {e}") from e

    try:
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        raise AsteroidModelError(f"Failed to download lightcurve from {url}: {e}") from e

    destination.write_bytes(response.content)
    return str(destination)
