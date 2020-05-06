BASE=../../..

EXECUTOR_PORT ?= 8000
LOCALHOST_PORT ?= 9000



run_executor_single:
	go run ${BASE}/main.go --sdep seldon-model --namespace default --predictor example --file ./deploy.yaml --http_port 8000

run_executor_chain:
	go run ${BASE}/main.go --sdep seldon-model --namespace default --predictor example --file ./chain.yaml --http_port 8000


run_mock_model:
	PREDICTIVE_UNIT_SERVICE_PORT=${LOCALHOST_PORT} seldon-core-microservice Model REST --service-type MODEL --log-level DEBUG


request:
	curl -s curl -X POST -H 'Content-Type: application/json' \
		 -d '{"data": {"names": ["input"], "ndarray": ["data"]}}' \
		 http://localhost:${EXECUTOR_PORT}/api/v1.0/predictions | jq .


metadata:
	curl -s http://localhost:${EXECUTOR_PORT}/api/v1.0/metadata/mock-model | jq .


check-metadata:
	curl -s http://localhost:${LOCALHOST_PORT}/metadata | jq .
