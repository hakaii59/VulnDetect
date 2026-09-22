"""Tests for the dataset builder — above all, that the split cannot leak.

The leakage these guard against is silent: it does not crash, it just makes
every downstream metric look better than it is. So the guarantee gets a test.

These need pandas + scikit-learn (requirements-dev.txt), but never the network:
`load_raw` is the only part that touches the Hub and is not exercised here.
"""

from __future__ import annotations

import pandas as pd
import pytest

from scripts.build_dataset import (
    GROUP_COL,
    class_weights,
    clean,
    deduplicate,
    split,
    verify,
)


def make_dataset(n_commits: int = 120, per_commit: int = 12) -> pd.DataFrame:
    """Synthetic Big-Vul-shaped data: many functions sharing few commits."""
    rows = []
    for c in range(n_commits):
        for f in range(per_commit):
            # Roughly 5% vulnerable, concentrated in some commits, as in Big-Vul.
            vul = 1 if (c % 7 == 0 and f < 4) else 0
            rows.append(
                {
                    "project": f"proj{c % 5}",
                    "commit_id": f"commit{c:04d}",
                    "CWE ID": "CWE-120" if vul else "",
                    "lang": "C" if c % 3 else "CPP",
                    "func_before": f"void fn_{c}_{f}(int x) {{ return x + {c * 100 + f}; }}",
                    "func_after": "void fn() {}",
                    "vul": vul,
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture()
def dataset():
    return deduplicate(clean(make_dataset()))


class TestClean:
    def test_normalizes_and_filters_language(self):
        df = clean(make_dataset())
        assert set(df["lang"]) == {"C", "C++"}, "CPP must normalize to C++"

    def test_drops_unknown_languages(self):
        raw = make_dataset()
        raw.loc[raw.index[:10], "lang"] = "Java"
        assert len(clean(raw)) == len(raw) - 10

    def test_drops_empty_functions(self):
        raw = make_dataset()
        raw.loc[raw.index[:5], "func_before"] = "   "
        assert len(clean(raw)) == len(raw) - 5

    def test_drops_rows_without_a_commit(self):
        raw = make_dataset()
        raw.loc[raw.index[:3], "commit_id"] = None
        assert len(clean(raw)) == len(raw) - 3


class TestDeduplicate:
    def test_collapses_whitespace_only_differences(self):
        """Exact-string dedup missed these; they were ~24% of Big-Vul."""
        raw = make_dataset(n_commits=12, per_commit=2)
        twin = raw.iloc[0].copy()
        twin["func_before"] = raw.iloc[0]["func_before"].replace(" ", "\n   ")
        twin["commit_id"] = "commit9999"
        padded = pd.concat([raw, twin.to_frame().T], ignore_index=True)

        assert padded["func_before"].nunique() == len(padded), "twin must differ as a raw string"
        assert len(deduplicate(clean(padded))) == len(raw)


class TestSplit:
    def test_proportions_are_80_10_10(self, dataset):
        splits = split(dataset, seed=42)
        total = len(dataset)
        assert sum(len(p) for p in splits.values()) == total
        assert len(splits["train"]) / total == pytest.approx(0.8, abs=0.03)
        assert len(splits["val"]) / total == pytest.approx(0.1, abs=0.03)
        assert len(splits["test"]) / total == pytest.approx(0.1, abs=0.03)

    def test_no_commit_spans_two_splits(self, dataset):
        """The whole point: a commit is indivisible."""
        splits = split(dataset, seed=42)
        groups = {name: set(part[GROUP_COL]) for name, part in splits.items()}
        assert not (groups["train"] & groups["test"])
        assert not (groups["train"] & groups["val"])
        assert not (groups["val"] & groups["test"])

    def test_label_rate_is_preserved(self, dataset):
        splits = split(dataset, seed=42)
        overall = dataset["vul"].mean()
        for name, part in splits.items():
            assert part["vul"].mean() == pytest.approx(overall, abs=0.02), name

    def test_is_deterministic_for_a_seed(self, dataset):
        a = split(dataset, seed=42)["test"][GROUP_COL].tolist()
        b = split(dataset, seed=42)["test"][GROUP_COL].tolist()
        assert a == b

    def test_seed_actually_changes_the_split(self, dataset):
        a = set(split(dataset, seed=42)["test"][GROUP_COL])
        b = set(split(dataset, seed=7)["test"][GROUP_COL])
        assert a != b


class TestProjectSplit:
    """The stricter, cross-codebase split (--group-by project)."""

    @staticmethod
    def lopsided(n_small: int = 60) -> pd.DataFrame:
        """One project dominating the rest, as Big-Vul's Chrome does (40% of rows)."""
        rows = []
        for f in range(800):
            rows.append({"project": "giant", "commit_id": f"c{f // 10}", "CWE ID": "",
                         "lang": "C", "func_before": f"void giant_{f}() {{ return; }}",
                         "func_after": "", "vul": int(f % 20 == 0)})
        for pnum in range(n_small):
            for f in range(10):
                rows.append({"project": f"small{pnum}", "commit_id": f"s{pnum}_{f // 5}", "CWE ID": "",
                             "lang": "C", "func_before": f"void small_{pnum}_{f}() {{ return; }}",
                             "func_after": "", "vul": int(f % 20 == 0)})
        return deduplicate(clean(pd.DataFrame(rows)))

    def test_no_project_spans_two_splits(self):
        df = self.lopsided()
        splits = split(df, seed=42, group_col="project")
        groups = {n: set(p["project"]) for n, p in splits.items()}
        assert not (groups["train"] & groups["test"])
        assert not (groups["train"] & groups["val"])
        assert not (groups["val"] & groups["test"])

    def test_a_dominant_project_does_not_wreck_the_proportions(self):
        """StratifiedGroupKFold would put the giant in one 10% fold; greedy
        assignment puts it in train and builds val/test from the tail."""
        df = self.lopsided()
        splits = split(df, seed=42, group_col="project")
        total = len(df)
        assert len(splits["train"]) / total == pytest.approx(0.8, abs=0.08)
        assert len(splits["val"]) / total == pytest.approx(0.1, abs=0.08)
        assert len(splits["test"]) / total == pytest.approx(0.1, abs=0.08)
        assert "giant" in set(splits["train"]["project"])

    def test_every_row_is_kept_exactly_once(self):
        df = self.lopsided()
        splits = split(df, seed=42, group_col="project")
        assert sum(len(p) for p in splits.values()) == len(df)

    def test_verify_checks_the_requested_group(self):
        df = self.lopsided()
        splits = split(df, seed=42, group_col="project")
        checks = verify(splits, group_col="project")
        assert set(checks.values()) == {0}
        assert any("project_overlap" in k for k in checks)

    def test_verify_raises_when_a_project_spans_splits(self):
        df = self.lopsided()
        splits = split(df, seed=42, group_col="project")
        leaked = splits["train"].iloc[[0]]
        splits["test"] = pd.concat([splits["test"], leaked], ignore_index=True)
        with pytest.raises(AssertionError, match="Leakage detected"):
            verify(splits, group_col="project")


class TestVerify:
    def test_passes_on_a_clean_split(self, dataset):
        checks = verify(split(dataset, seed=42))
        assert set(checks.values()) == {0}

    def test_raises_when_a_commit_spans_splits(self, dataset):
        """A regression that reintroduces leakage must fail loudly."""
        splits = split(dataset, seed=42)
        leaked = splits["train"].iloc[[0]]
        splits["test"] = pd.concat([splits["test"], leaked], ignore_index=True)

        with pytest.raises(AssertionError, match="Leakage detected"):
            verify(splits)

    def test_raises_when_code_is_duplicated_across_splits(self, dataset):
        splits = split(dataset, seed=42)
        borrowed = splits["train"].iloc[[0]].copy()
        borrowed[GROUP_COL] = "commit-from-nowhere"  # different commit, same code
        splits["test"] = pd.concat([splits["test"], borrowed], ignore_index=True)

        with pytest.raises(AssertionError, match="Leakage detected"):
            verify(splits)


class TestClassWeights:
    def test_rarer_class_gets_the_larger_weight(self, dataset):
        train = split(dataset, seed=42)["train"]
        weights = class_weights(train)
        assert weights["1"] > weights["0"]

    def test_weights_balance_the_classes(self, dataset):
        train = split(dataset, seed=42)["train"]
        weights = class_weights(train)
        counts = train["vul"].value_counts()
        assert counts[0] * weights["0"] == pytest.approx(counts[1] * weights["1"])
