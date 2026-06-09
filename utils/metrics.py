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


class CampaignOptMonitoringClient(GoogleAdsMetricsClient):
    """Optional Grafana emission for production plan-vs-actual monitoring."""

    def emit_production_monitoring_metrics(
        self,
        course: str,
        score_date: str,
        metrics: dict[str, float | None],
    ) -> None:
        labels = {"course": course, "score_date": score_date}
        prefix = "campaign_opt_monitoring"
        field_map = {
            "rmse": metrics.get("rmse_pred_vs_observed"),
            "nrmse": metrics.get("nrmse"),
            "bias_pct": metrics.get("total_bias_pct"),
            "pred_total": metrics.get("pred_total"),
            "observed_total": metrics.get("observed_total"),
        }
        for field_name, value in field_map.items():
            if value is not None:
                self.emit_metric(field_name, float(value), labels, metric_prefix=prefix)


def get_metrics_client() -> CampaignOptMonitoringClient:
    return CampaignOptMonitoringClient(
        os.getenv("GRAFANA_URL"),
        os.getenv("GRAFANA_USERNAME"),
        os.getenv("GRAFANA_TOKEN"),
    )


google_ads_metrics_client = get_metrics_client()
