from unittest.mock import patch

from fastapi.testclient import TestClient

from app.main import app
from evaluation.report import build_report

client = TestClient(app)


class TestGetEvaluation:
    def test_returns_404_when_no_report_exists(self):
        with patch("app.routers.evaluation.load_latest_report", return_value=None):
            response = client.get("/api/evaluation")

        assert response.status_code == 404

    def test_returns_latest_report_when_present(self):
        report = build_report([], used_judge=False)

        with patch("app.routers.evaluation.load_latest_report", return_value=report):
            response = client.get("/api/evaluation")

        assert response.status_code == 200
        body = response.json()
        assert body["total"] == 0
        assert "results" in body
