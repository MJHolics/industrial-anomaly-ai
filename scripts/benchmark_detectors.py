"""이상탐지기 정직한 head-to-head 비교 + 원칙적 임계(operating-point) 선택.

현재 서빙은 IsolationForest 단일 + 하드코딩 임계(score>-0.15)다. 그 선택이 *옳은지*를
막연히 주장하지 않고 **측정으로 자리 정한다**(D9식). 5개 탐지기를 같은 프로토콜로 재고,
임계는 test가 아니라 **validation에서** 골라(누수 차단) test에 보고한다.

프로토콜(semi-supervised novelty detection — 실무의 "정상 가동 구간으로 학습"과 정합):
  - 시간순 분할(누수 방지): train 60% / val 20% / test 20%.
  - 학습은 train의 **정상(label=0)만** 사용(비지도). 라벨은 *평가·임계선택*에만 쓴다.
  - 지표: ROC-AUC·PR-AUC(평균정밀도) = **임계 무관**(불균형 4.8%에 정직). 운영점 F1은 val에서 임계 적합.

탐지기: IsolationForest · LocalOutlierFactor(novelty) · OneClassSVM · EllipticEnvelope(로버스트
가우시안) · RobustZ(중앙값/MAD max-z 베이스라인). 모두 "높을수록 이상"으로 부호 정렬.

핵심 헬퍼(임계 선택·지표)는 순수 numpy → 합성 데이터로 단위검증(`tests/test_benchmark.py`).

사용:
    python scripts/benchmark_detectors.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

DATA = ROOT / "data" / "raw" / "sensor_data.csv"
OUT_DIR = ROOT / "data" / "results"
FEATURES = ["temperature", "vibration", "current"]
SEED = 42


# ---------------------------------------------------------------------------
# 순수 헬퍼 (네트워크·모델 무관, 단위 검증 가능)
# ---------------------------------------------------------------------------
def best_f1_threshold(scores: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    """후보 임계(정렬된 점수)를 훑어 F1 최대 임계와 그 F1을 반환. score>=t → 이상(1).

    비지도 점수를 운영점으로 바꾸는 결정적 규칙. val에서 골라 test에 적용(누수 차단).
    """
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=int)
    order = np.unique(scores)
    if len(order) == 0:
        return (0.0, 0.0)
    best_t, best_f1 = order[0], -1.0
    P = labels.sum()
    for t in order:
        pred = scores >= t
        tp = int((pred & (labels == 1)).sum())
        fp = int((pred & (labels == 0)).sum())
        if tp == 0:
            continue
        prec = tp / (tp + fp)
        rec = tp / P if P else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        if f1 > best_f1:
            best_f1, best_t = f1, float(t)
    return (best_t, float(best_f1))


def prf_at_threshold(scores: np.ndarray, labels: np.ndarray, t: float) -> dict:
    """임계 t에서 precision/recall/F1 (score>=t → 이상)."""
    pred = np.asarray(scores) >= t
    labels = np.asarray(labels, dtype=int)
    tp = int((pred & (labels == 1)).sum())
    fp = int((pred & (labels == 0)).sum())
    fn = int((~pred & (labels == 1)).sum())
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return {"precision": round(prec, 4), "recall": round(rec, 4), "f1": round(f1, 4),
            "tp": tp, "fp": fp, "fn": fn}


# ---------------------------------------------------------------------------
# 탐지기들 — fit(정상만) → score(높을수록 이상)
# ---------------------------------------------------------------------------
def _build_detectors():
    from sklearn.covariance import EllipticEnvelope
    from sklearn.ensemble import IsolationForest
    from sklearn.neighbors import LocalOutlierFactor
    from sklearn.svm import OneClassSVM

    return {
        "IsolationForest": lambda: IsolationForest(n_estimators=200, random_state=SEED, n_jobs=-1),
        "LocalOutlierFactor": lambda: LocalOutlierFactor(n_neighbors=20, novelty=True),
        "OneClassSVM": lambda: OneClassSVM(nu=0.05, gamma="scale"),
        "EllipticEnvelope": lambda: EllipticEnvelope(contamination=0.05, random_state=SEED),
    }


class RobustZ:
    """중앙값/MAD 기반 max robust-z 베이스라인(순수 numpy) — 단순·빠름·설명가능."""

    def fit(self, X):
        self.med_ = np.median(X, axis=0)
        mad = np.median(np.abs(X - self.med_), axis=0)
        self.mad_ = np.where(mad < 1e-9, 1e-9, mad) * 1.4826  # 정규분포 일치 상수
        return self

    def score_samples(self, X):  # 높을수록 정상이 되도록 sklearn과 부호 맞춤(-anomaly)
        z = np.abs((X - self.med_) / self.mad_).max(axis=1)
        return -z


def _anomaly_scores(det, X):
    """sklearn score_samples는 '높을수록 정상' → 이상점수 = 음수화(높을수록 이상)."""
    return -det.score_samples(X)


def main() -> None:
    from sklearn.metrics import average_precision_score, roc_auc_score
    from sklearn.preprocessing import StandardScaler

    df = pd.read_csv(DATA).sort_values("timestamp").reset_index(drop=True)
    X = df[FEATURES].to_numpy(float)
    y = df["label"].to_numpy(int)
    n = len(df)
    i_tr, i_va = int(n * 0.6), int(n * 0.8)
    Xtr, ytr = X[:i_tr], y[:i_tr]
    Xva, yva = X[i_tr:i_va], y[i_tr:i_va]
    Xte, yte = X[i_va:], y[i_va:]
    print(f"[data] n={n} · train {len(ytr)}(정상만 학습) · val {len(yva)}(이상 {yva.sum()}) "
          f"· test {len(yte)}(이상 {yte.sum()})")

    scaler = StandardScaler().fit(Xtr[ytr == 0])   # 정상만으로 스케일 적합
    Xtr_n = scaler.transform(Xtr[ytr == 0])
    Xva_s, Xte_s = scaler.transform(Xva), scaler.transform(Xte)

    builders = dict(_build_detectors())
    builders["RobustZ(baseline)"] = lambda: RobustZ()

    results = {}
    for name, build in builders.items():
        det = build().fit(Xtr_n)
        sv = _anomaly_scores(det, Xva_s)
        st = _anomaly_scores(det, Xte_s)
        roc = float(roc_auc_score(yte, st))
        pr = float(average_precision_score(yte, st))
        t, f1_val = best_f1_threshold(sv, yva)       # 임계는 val에서 선택
        test_pt = prf_at_threshold(st, yte, t)        # test에 적용해 보고
        results[name] = {"roc_auc": round(roc, 4), "pr_auc": round(pr, 4),
                         "val_f1": round(f1_val, 4), "threshold": round(t, 4),
                         "test": test_pt}

    order = sorted(results, key=lambda k: results[k]["pr_auc"], reverse=True)
    print("\n=== 탐지기 head-to-head (test, 임계는 val에서 선택 → 누수 0) ===")
    print(f"{'detector':22s} {'ROC-AUC':>8s} {'PR-AUC':>8s} | test  P / R / F1")
    for name in order:
        r = results[name]; t = r["test"]
        print(f"{name:22s} {r['roc_auc']:8.3f} {r['pr_auc']:8.3f} | "
              f"{t['precision']:.3f} / {t['recall']:.3f} / {t['f1']:.3f}")

    # 대조: 현재 서빙의 하드코딩 임계(IsolationForest, score>-0.15 = 이상점수>0.15) test 성능.
    det = builders["IsolationForest"]().fit(Xtr_n)
    st = _anomaly_scores(det, Xte_s)
    hard = prf_at_threshold(st, yte, 0.15)
    best = results["IsolationForest"]["test"]
    print(f"\n[대조] IsolationForest 하드코딩 임계(0.15) → P{hard['precision']:.3f}/R{hard['recall']:.3f}/"
          f"F1 {hard['f1']:.3f}  vs  val-선택 임계 → F1 {best['f1']:.3f} "
          f"({'val 선택이 우위' if best['f1'] >= hard['f1'] else '하드코딩이 우위'})")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / "detector_benchmark.json"
    out.write_text(json.dumps({
        "n": n, "split": {"train": len(ytr), "val": len(yva), "test": len(yte)},
        "anomaly_rate": round(float(y.mean()), 4),
        "protocol": "semi-supervised novelty: fit on train-normal only; threshold on val; report on test",
        "ranked_by": "pr_auc", "results": results,
        "hardcoded_threshold_ref": {"threshold": 0.15, "test": hard},
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n결과: {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
