import os
import traceback

import requests

REQUEST_HEADERS = {"Content-Type": "text/plain"}


class MetricsClient:
    """Barebones client for optionally emitting Influx line-format metrics."""

    def __init__(self, url: str | None, username: str | None, token: str | None) -> None:
        self.url = url
        self.username = username
        self.token = token
        self.metric_prefix = "google_ads_data_pull"

    def emit_metric(
        self,
        metric_name: str,
        value: float,
        labels: dict[str, str],
        metric_prefix: str | None = None,
    ) -> None:
        if not all([self.url, self.username, self.token]):
            return

        try:
            prefix = metric_prefix or self.metric_prefix
            labels_string = ",".join([f"{key}={value}" for key, value in labels.items()])
            payload = f"{prefix},{labels_string} {metric_name}={value}"
            response = requests.post(
                self.url,
                headers=REQUEST_HEADERS,
                data=payload,
                auth=(self.username, self.token),
                timeout=10,
            )
            response.raise_for_status()
        except Exception:
            print(f"Error emitting metric {metric_name} with value {value} and labels {labels}")
            print(traceback.format_exc())


class GoogleAdsMetricsClient(MetricsClient):
    def track_google_ads_operation_count(self, operation_type: str, value: int) -> None:
        self.emit_metric("api_operation_count", float(value), {"operation": operation_type})


def get_metrics_client() -> GoogleAdsMetricsClient:
    return GoogleAdsMetricsClient(
        os.getenv("GRAFANA_URL"),
        os.getenv("GRAFANA_USERNAME"),
        os.getenv("GRAFANA_TOKEN"),
    )


google_ads_metrics_client = get_metrics_client()
