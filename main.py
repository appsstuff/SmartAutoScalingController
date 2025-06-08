import asyncio
import os
from async_fetch_services import run_all_services
from prometheus_ml_preprocessor import preprocess_metrics
from model_inference import run_gpr_model, run_xgb_model, run_lstm_model
from autoscale_fusion import fuse_predictions
from k8s_scaler import apply_k8s_scaling
from shared_config import AUTO_SCALE_SERVICES


def main():
    print("\U0001F680 Smart Autoscaler starting...")

    loop = asyncio.get_event_loop()
    metrics_list = loop.run_until_complete(run_all_services())

    for svc, raw_metrics in zip(AUTO_SCALE_SERVICES, metrics_list):
        print(f"\n📦 Processing service: {svc['pod_name']}")

        try:
            # 1. Preprocess metrics → ML features
            features = preprocess_metrics(raw_metrics)

            # 2. Run models (GPR, LSTM, XGB)
            gpr_class = run_gpr_model(features)
            xgb_pred = run_xgb_model(features)
            lstm_class = run_lstm_model(features)

            # 3. Fuse outputs → scaling decision
            decision = fuse_predictions(gpr_class, xgb_pred, lstm_class)

            # 4. Apply scaling
            apply_k8s_scaling(
                pod_name=svc["pod_name"],
                namespace=svc["namespace"],
                decision=decision
            )

        except Exception as e:
            print(f"❌ Error while processing {svc['pod_name']}: {e}")

    print("\n✅ Autoscaler cycle complete.")

if __name__ == "__main__":
    main()
