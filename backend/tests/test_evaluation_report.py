from evaluation.report import (
    EvaluationResult,
    MetricScore,
    build_report,
    load_latest_report,
    save_report,
)


def _result(name: str, passed: bool, score: float, category: str = "cat_a") -> EvaluationResult:
    return EvaluationResult(
        test_name=name,
        category=category,
        is_red_team=False,
        passed=passed,
        overall_score=score,
        metrics=[MetricScore(name="uncertainty", applicable=True, passed=passed, score=score)],
        failure_reasons=[] if passed else ["uncertainty: too confident"],
    )


class TestBuildReport:
    def test_counts_and_averages(self):
        results = [
            _result("a", True, 1.0),
            _result("b", False, 0.2, category="cat_b"),
            _result("c", True, 0.8),
        ]
        report = build_report(results, used_judge=False)

        assert report.total == 3
        assert report.passed == 2
        assert report.failed == 1
        assert report.overall_score == round((1.0 + 0.2 + 0.8) / 3, 3)
        assert report.metric_averages["uncertainty"] == round((1.0 + 0.2 + 0.8) / 3, 3)
        assert report.failure_categories == {"cat_b": 1}

    def test_empty_results_do_not_crash(self):
        report = build_report([], used_judge=False)
        assert report.total == 0
        assert report.overall_score == 0.0


class TestSaveAndLoadReport:
    def test_roundtrip(self, tmp_path, monkeypatch):
        import evaluation.report as report_module

        monkeypatch.setattr(report_module, "REPORTS_DIR", tmp_path)
        monkeypatch.setattr(report_module, "LATEST_REPORT_PATH", tmp_path / "latest.json")

        report = build_report([_result("a", True, 1.0)], used_judge=True)
        report_module.save_report(report)

        loaded = report_module.load_latest_report()
        assert loaded is not None
        assert loaded.total == 1
        assert loaded.results[0].test_name == "a"
        assert loaded.used_judge is True

    def test_returns_none_when_no_report_exists(self, tmp_path, monkeypatch):
        import evaluation.report as report_module

        monkeypatch.setattr(report_module, "LATEST_REPORT_PATH", tmp_path / "missing.json")
        assert report_module.load_latest_report() is None
