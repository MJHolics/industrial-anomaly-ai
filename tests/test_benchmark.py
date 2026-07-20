"""이상탐지 벤치마크 핵심(임계 선택·지표·RobustZ) 단위테스트 — 순수, 데이터/모델 불요.

불변식:
  1) best_f1_threshold는 분리 가능한 점수에서 이상/정상을 가르는 임계를 찾는다.
  2) prf_at_threshold의 P/R/F1이 손계산과 일치.
  3) RobustZ가 정상 대비 이상표본에 더 높은 이상점수를 준다.
  4) val에서 고른 임계가 유사분포 test에 일반화(누수 없이도 동작).
"""
from __future__ import annotations

import numpy as np

from scripts.benchmark_detectors import RobustZ, best_f1_threshold, prf_at_threshold


def test_best_f1_threshold_separable():
    # 정상 점수 낮고 이상 점수 높음 → 사이 임계에서 F1=1.
    scores = np.array([0.1, 0.2, 0.15, 0.9, 0.95, 0.8])
    labels = np.array([0, 0, 0, 1, 1, 1])
    t, f1 = best_f1_threshold(scores, labels)
    assert f1 == 1.0
    assert 0.2 < t <= 0.8


def test_prf_at_threshold_matches_handcount():
    scores = np.array([0.9, 0.1, 0.8, 0.2])
    labels = np.array([1, 0, 1, 1])          # 3 anomalies
    r = prf_at_threshold(scores, labels, 0.5)   # pred anomaly: idx0,2
    assert r["tp"] == 2 and r["fp"] == 0 and r["fn"] == 1
    assert r["precision"] == 1.0
    assert round(r["recall"], 3) == round(2 / 3, 3)


def test_robustz_ranks_anomalies_higher():
    rng = np.random.default_rng(0)
    normal = rng.normal(0, 1, size=(500, 3))
    det = RobustZ().fit(normal)
    anomalies = np.array([[8.0, 8.0, 8.0], [-9.0, 0.0, 7.0]])
    s_normal = -det.score_samples(normal)        # 이상점수(높을수록 이상)
    s_anom = -det.score_samples(anomalies)
    assert s_anom.min() > np.percentile(s_normal, 99)


def test_val_selected_threshold_generalizes():
    rng = np.random.default_rng(1)
    def make(n_norm, n_anom):
        s = np.concatenate([rng.normal(0.2, 0.05, n_norm), rng.normal(0.9, 0.05, n_anom)])
        y = np.concatenate([np.zeros(n_norm), np.ones(n_anom)]).astype(int)
        return s, y
    sv, yv = make(400, 20)
    st, yt = make(400, 20)
    t, _ = best_f1_threshold(sv, yv)
    r = prf_at_threshold(st, yt, t)             # val 임계 → test
    assert r["f1"] > 0.9                         # 누수 없이 일반화
