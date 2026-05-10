"""
MQTT 센서 시뮬레이터.
정상 / 이상 패턴 데이터를 브로커에 publish.

실행: python scripts/mqtt_simulator.py [--anomaly]
  --anomaly 플래그를 붙이면 이상 패턴 데이터 전송
"""

import json
import time
import argparse
import numpy as np
from datetime import datetime

try:
    import paho.mqtt.client as mqtt
    MQTT_AVAILABLE = True
except ImportError:
    MQTT_AVAILABLE = False
    print("[경고] paho-mqtt가 없습니다. pip install paho-mqtt")

BROKER_HOST = "localhost"
BROKER_PORT = 1883
TOPIC = "sensors/factory/line1"
INTERVAL = 1.0  # 초
SENSOR_ID = "sensor_01"

rng = np.random.RandomState()


def make_normal_reading() -> dict:
    return {
        "sensor_id": SENSOR_ID,
        "temperature": round(float(rng.normal(70, 3)), 2),
        "vibration":   round(float(rng.normal(0.5, 0.08)), 3),
        "current":     round(float(rng.normal(12, 0.8)), 2),
        "timestamp":   datetime.now().isoformat(),
    }


def make_anomaly_reading(anomaly_type: str = "random") -> dict:
    if anomaly_type == "random":
        anomaly_type = rng.choice(["overheating", "vibration", "current"])

    reading = make_normal_reading()
    if anomaly_type == "overheating":
        reading["temperature"] = round(float(rng.normal(95, 5)), 2)
    elif anomaly_type == "vibration":
        reading["vibration"] = round(float(rng.normal(2.5, 0.3)), 3)
    elif anomaly_type == "current":
        reading["current"] = round(float(rng.normal(30, 3)), 2)
    return reading


def run(send_anomaly: bool = False):
    if not MQTT_AVAILABLE:
        print("MQTT 없이 콘솔 출력 모드로 실행합니다.")
        while True:
            data = make_anomaly_reading() if send_anomaly else make_normal_reading()
            label = "[이상]" if send_anomaly else "[정상]"
            print(f"{label} {json.dumps(data, ensure_ascii=False)}")
            time.sleep(INTERVAL)
        return

    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)

    def on_connect(c, userdata, flags, reason_code, properties):
        if reason_code == 0:
            print(f"브로커 연결 성공: {BROKER_HOST}:{BROKER_PORT}")
        else:
            print(f"연결 실패 코드: {reason_code}")

    client.on_connect = on_connect
    client.connect(BROKER_HOST, BROKER_PORT, 60)
    client.loop_start()

    print(f"토픽 [{TOPIC}] 으로 데이터 전송 시작 ({'이상' if send_anomaly else '정상'} 패턴)")
    try:
        while True:
            data = make_anomaly_reading() if send_anomaly else make_normal_reading()
            client.publish(TOPIC, json.dumps(data))
            label = "[이상]" if send_anomaly else "[정상]"
            print(f"{label} {json.dumps(data, ensure_ascii=False)}")
            time.sleep(INTERVAL)
    except KeyboardInterrupt:
        print("\n시뮬레이터 종료.")
        client.loop_stop()
        client.disconnect()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--anomaly", action="store_true", help="이상 패턴 전송")
    args = parser.parse_args()
    run(send_anomaly=args.anomaly)
