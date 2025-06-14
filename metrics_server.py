from flask import Flask
from prometheus_client import make_wsgi_app
from werkzeug.middleware.dispatcher import DispatcherMiddleware
from prometheus_client import Gauge, Enum
from threading import Thread

# Define custom metrics
DECISION_GAUGE = Gauge('autoscaler_decision', 'Last autoscaler decision', ['service'])
CPU_USAGE_GAUGE = Gauge('autoscaler_cpu_usage', 'Current CPU usage from feature vector', ['service'])
MEM_USAGE_GAUGE = Gauge('autoscaler_mem_usage', 'Current memory usage (MB)', ['service'])
REQ_RATE_GAUGE = Gauge('autoscaler_req_rate', 'Requests per second', ['service'])


def set_metric_values(service, raw_metrics, decision_value):
    DECISION_GAUGE.labels(service=service).set(decision_value)
    CPU_USAGE_GAUGE.labels(service=service).set(raw_metrics["cpu_usage_lag_1"])
    MEM_USAGE_GAUGE.labels(service=service).set(raw_metrics["mem_usage"])
    REQ_RATE_GAUGE.labels(service=service).set(raw_metrics["req_rate"])


app = Flask(__name__)
app.wsgi_app = DispatcherMiddleware(app.wsgi_app, {
    '/metrics': make_wsgi_app()
})


@app.route('/healthz')
def health_check():
    return {"status": "healthy"}, 200


@app.route('/ready')
def ready_check():
    from autoscaler_controller import history_ready
    return {"status": "healthy" if history_ready else "unhealthy"}, 200 if history_ready else 503


def run_metrics_server(port=8900):
    print(f"📊 Starting Prometheus metrics server on port {port}")
    app.run(host='0.0.0.0', port=port, threaded=True)


def start_metrics_server(port=8900):
    thread = Thread(target=run_metrics_server, args=(port,))
    thread.daemon = True
    thread.start()