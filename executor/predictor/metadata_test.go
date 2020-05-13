package predictor

import (
	"encoding/json"
	. "github.com/onsi/gomega"
	"github.com/seldonio/seldon-core/operator/apis/machinelearning.seldon.io/v1"
	"testing"
)

func TestSingleModel(t *testing.T) {
	t.Logf("Started")
	g := NewGomegaWithT(t)
	model := v1.MODEL

	spec := &v1.PredictorSpec{
		Name: "predictor-name",
		Graph: &v1.PredictiveUnit{
			Name: "model-1",
			Type: &model,
			Endpoint: &v1.Endpoint{
				ServiceHost: "foo",
				ServicePort: 9000,
				Type:        v1.REST,
			},
		},
	}
	p := createPredictorProcess(t)

	graphMetadata, err := NewGraphMetadata(p, spec)
	g.Expect(err).Should(BeNil())

	expectedGrahMetadata := GraphMetadata{
		Name: "predictor-name",
		Models: map[string]ModelMetadata{
			"model-1": {
				Name:     "model-1",
				Platform: "platform-name",
				Versions: []string{"model-version"},
				Inputs: []MetadataTensor{
					{Name: "input", DataType: "BYTES", Shape: []int{1, 5}},
				},
				Outputs: []MetadataTensor{
					{Name: "output", DataType: "BYTES", Shape: []int{1, 3}},
				},
			},
		},
		GraphInputs: []MetadataTensor{
			{Name: "input", DataType: "BYTES", Shape: []int{1, 5}},
		},
		GraphOutputs: []MetadataTensor{
			{Name: "output", DataType: "BYTES", Shape: []int{1, 3}},
		},
	}
	g.Expect(*graphMetadata).To(Equal(expectedGrahMetadata))
}

func TestChainModel(t *testing.T) {
	t.Logf("Started")
	g := NewGomegaWithT(t)
	p := createPredictorProcess(t)

	var specJson = `{
		"name": "predictor-name",
		"graph": {
		   "name": "model-1",
		   "type": "MODEL",
		   "endpoint": {
		        "service_host": "localhost",
		        "service_port": 9000
		    },
		    "children": [
		        {
		           "name": "model-2",
		           "type": "MODEL",
		           "endpoint": {
		                "service_host": "localhost",
		                "service_port": 9001
		            }
		        }
		    ]
		}
	}`

	var graphMetaJson = `{
		"Name": "predictor-name",
		"Models": {
			"model-1": {
				"Name": "model-1",
				"versions": ["model-version"],
				"Platform": "platform-name",
				"Inputs": [{"DataType": "BYTES", "Name": "input", "Shape": [1, 5]}],
				"Outputs": [{"DataType": "BYTES", "Name": "output", "Shape": [1, 3]}]
			},
			"model-2": {
				"Name": "model-2",
				"versions": ["model-version"],
				"Platform": "platform-name",
				"Inputs": [{"DataType": "BYTES", "Name": "input", "Shape": [1, 3]}],
				"Outputs": [{"DataType": "BYTES", "Name": "output", "Shape": [3]}]
			}
		},
		"GraphInputs": [{"DataType": "BYTES", "Name": "input", "Shape": [1, 5]}],
		"GraphOutputs": [{"DataType": "BYTES", "Name": "output", "Shape": [3]}]
	}`

	var expectedGrahMetadata GraphMetadata
	err := json.Unmarshal([]byte(graphMetaJson), &expectedGrahMetadata)
	g.Expect(err).Should(BeNil())

	var spec v1.PredictorSpec
	err = json.Unmarshal([]byte(specJson), &spec)
	g.Expect(err).Should(BeNil())

	graphMetadata, err := NewGraphMetadata(p, &spec)
	g.Expect(err).Should(BeNil())

	g.Expect(*graphMetadata).To(Equal(expectedGrahMetadata))
}

func TestCombinerModel(t *testing.T) {
	t.Logf("Started")
	g := NewGomegaWithT(t)
	p := createPredictorProcess(t)

	var specJson = `{
		"name": "predictor-name",
		"graph": {
		   "name": "model-combiner",
		   "type": "COMBINER",
		   "endpoint": {
		        "service_host": "localhost",
		        "service_port": 9000
		    },
		    "children": [
		        {
		           "name": "model-a1",
		           "type": "MODEL",
		           "endpoint": {
		                "service_host": "localhost",
		                "service_port": 9001
		            }
		        },
		        {
		           "name": "model-a2",
		           "type": "MODEL",
		           "endpoint": {
		                "service_host": "localhost",
		                "service_port": 9002
		            }
		        }
		    ]
		}
	}`

	var graphMetaJson = `{
		"Name": "predictor-name",
		"Models": {
			"model-a1": {
				"Name": "model-a1",
				"versions": ["model-version"],
				"Platform": "platform-name",
				"Inputs": [{"DataType": "BYTES", "Name": "input", "Shape": [1, 5]}],
				"Outputs": [{"DataType": "BYTES", "Name": "output", "Shape": [1, 10]}]
			},
			"model-a2": {
				"Name": "model-a2",
				"versions": ["model-version"],
				"Platform": "platform-name",
				"Inputs": [{"DataType": "BYTES", "Name": "input", "Shape": [1, 5]}],
				"Outputs": [{"DataType": "BYTES", "Name": "output", "Shape": [1, 20]}]
			},
		    "model-combiner": {
		        "name": "model-combiner",
		        "versions": ["model-version"],
		        "platform": "platform-name",
		        "inputs": [
		            {"name": "input-1", "datatype": "BYTES", "shape": [1, 10]},
		            {"name": "input-2", "datatype": "BYTES", "shape": [1, 20]}
		        ],
		        "outputs": [{"name": "combined output", "datatype": "BYTES", "shape": [3]}]
		    }
		},
		"GraphInputs": [{"DataType": "BYTES", "Name": "input", "Shape": [1, 5]}],
		"GraphOutputs": [{"DataType": "BYTES", "Name": "combined output", "Shape": [3]}]
	}`

	var expectedGrahMetadata GraphMetadata
	err := json.Unmarshal([]byte(graphMetaJson), &expectedGrahMetadata)
	g.Expect(err).Should(BeNil())

	var spec v1.PredictorSpec
	err = json.Unmarshal([]byte(specJson), &spec)
	g.Expect(err).Should(BeNil())

	graphMetadata, err := NewGraphMetadata(p, &spec)
	g.Expect(err).Should(BeNil())

	g.Expect(*graphMetadata).To(Equal(expectedGrahMetadata))
}
