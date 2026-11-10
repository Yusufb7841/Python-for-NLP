import pandas as pd

from agnews.experiments import SUITES, RunSpec, append_rows, completed_keys, expand_suites


def test_tfidf_runs_once_per_config_regardless_of_embedding_axis():
    specs = expand_suites(["main"], seeds=[0, 1])
    tfidf = [s for s in specs if s.model == "tfidf"]
    assert len(tfidf) == 2 and {s.embeddings for s in tfidf} == {"none"}
    neural = [s for s in specs if s.model != "tfidf"]
    assert len(neural) == 4 * len(SUITES["main"]["embeddings"]) * 2


def test_run_keys_are_unique_across_suites():
    specs = expand_suites(list(SUITES), seeds=[0])
    assert len({s.key for s in specs}) == len(specs)


def test_model_filter():
    specs = expand_suites(["sequence_length"], seeds=[0], models=["fnn"])
    assert {s.model for s in specs} == {"fnn"}


def test_results_file_appends_and_reports_completed_keys(tmp_path):
    path = tmp_path / "runs.csv"
    spec = RunSpec("main", "fnn", "random", None, 64, 0)
    assert completed_keys(path) == set()
    append_rows(path, [{"key": spec.key, "test_accuracy": 0.9}])
    append_rows(path, [{"key": "other", "test_accuracy": 0.8}])
    assert completed_keys(path) == {spec.key, "other"}
    assert len(pd.read_csv(path)) == 2
