raw_metrics = fetch_pod_metrics(pod_name, namespace)
input_row = build_feature_vector(raw_metrics, feature_names)
decision = predict_scaling_action(input_row, seq_input, service_name=pod_name)
apply_k8s_scaling(pod_name, namespace, decision)