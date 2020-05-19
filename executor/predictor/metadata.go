package predictor

import (
	"github.com/seldonio/seldon-core/operator/apis/machinelearning.seldon.io/v1"
	"sigs.k8s.io/controller-runtime/pkg/runtime/log"
)

type MetadataTensor struct {
	DataType string `json:"datatype,omitempty"`
	Name     string `json:"name,omitempty"`
	Shape    []int  `json:"shape,omitempty"`
}

type ModelMetadata struct {
	Name     string           `json:"name,omitempty"`
	Platform string           `json:"platform,omitempty"`
	Versions []string         `json:"versions,omitempty"`
	Inputs   []MetadataTensor `json:"inputs,omitempty"`
	Outputs  []MetadataTensor `json:"outputs,omitempty"`
}

type GraphMetadata struct {
	Name         string                   `json:"name"`
	Models       map[string]ModelMetadata `json:"models"`
	GraphInputs  []MetadataTensor         `json:"graphinputs"`
	GraphOutputs []MetadataTensor         `json:"graphoutputs"`
}

func NewGraphMetadata(p *PredictorProcess, spec *v1.PredictorSpec) (output *GraphMetadata, err error) {
	output = &GraphMetadata{}
	output.Models, err = p.MetadataMap(spec.Graph)
	if err != nil {
		return nil, err
	}
	output.Name = spec.Name
	output.GraphInputs, output.GraphOutputs = output.GetShapeFromGraph(spec.Graph)
	return
}

func (p *GraphMetadata) GetShapeFromGraph(node *v1.PredictiveUnit) (
	input []MetadataTensor, output []MetadataTensor,
) {
	nodeMeta := p.Models[node.Name]
	nodeInputs := nodeMeta.Inputs
	nodeOutputs := nodeMeta.Outputs

	// Single node graphs: code path terminates here if this is the case
	if node.Children == nil || len(node.Children) == 0 {
		// We treat node's inputs/outputs as global despite its Type
		return nodeInputs, nodeOutputs
	}

	// Multi nodes graphs
	if *node.Type == v1.MODEL || *node.Type == v1.TRANSFORMER {
		// Ignore all children except first one for Models and Transformers
		_, childOutputs := p.GetShapeFromGraph(&node.Children[0])
		return nodeInputs, childOutputs
	} else if *node.Type == v1.OUTPUT_TRANSFORMER {
		// Ignore all children except first one for Output Transformers
		// OUTPUT_TRANSFORMER first passes its input to (first) child and returns the output.
		childInputs, _ := p.GetShapeFromGraph(&node.Children[0])
		return childInputs, nodeOutputs
	} else if *node.Type == v1.COMBINER {
		// Combiner will pass request to all of its children and combine their output.
		// We assume that all children take same type of inputs.
		childInputs, _ := p.GetShapeFromGraph(&node.Children[0])

		return childInputs, nodeOutputs
	} else if *node.Type == v1.ROUTER {
		// ROUTER will pass request to one of its children and return child's output.
		// We assume that all children take same type of inputs.
		childInputs, childOutputs := p.GetShapeFromGraph(&node.Children[0])
		return childInputs, childOutputs
	}

	// If we got here it means none of the cases above
	logger := log.Log.WithName("GraphMetadata")
	logger.Info("Unimplemented case: Couldn't derive graph-level inputs and outputs.")
	return nil, nil
}
