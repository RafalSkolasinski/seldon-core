package predictor

import (
	"fmt"
	"github.com/seldonio/seldon-core/operator/apis/machinelearning.seldon.io/v1"
)

type MetadataTensor struct {
	DataType string
	Name     string
	Shape    []int
}

type ModelMetadata struct {
	Name     string
	Platform string
	Inputs   []MetadataTensor
	Outputs  []MetadataTensor
}

type GraphMetadata struct {
	Name         string
	Models       map[string]ModelMetadata
	GraphInputs  []MetadataTensor
	GraphOutputs []MetadataTensor
}

type GraphMetadataPredictor struct {
	PredictorProcess *PredictorProcess
	PredictorSpec    *v1.PredictorSpec
	metadataMap      map[string]ModelMetadata
}

func NewGraphMetadataPredictor(p *PredictorProcess, spec *v1.PredictorSpec) GraphMetadataPredictor {
	return GraphMetadataPredictor{
		PredictorProcess: p,
		PredictorSpec:    spec,
	}
}

func (p *GraphMetadataPredictor) GraphMetadata() (output *GraphMetadata, err error) {
	p.metadataMap, err = p.PredictorProcess.MetadataMap(p.PredictorSpec.Graph)
	if err != nil {
		return nil, err
	}

	inp, out := p.GetShapeFromGraph(p.PredictorSpec.Graph)

	return &GraphMetadata{
		Name:         p.PredictorSpec.Name,
		Models:       p.metadataMap,
		GraphInputs:  inp,
		GraphOutputs: out,
	}, nil
}

func (p *GraphMetadataPredictor) GetShapeFromGraph(node *v1.PredictiveUnit) (
	input []MetadataTensor, output []MetadataTensor,
) {
	nodeMeta := p.metadataMap[node.Name]
	nodeInputs := nodeMeta.Inputs
	nodeOutputs := nodeMeta.Outputs

	// Node has no children
	if node.Children == nil || len(node.Children) == 0 {
		// If Node is model then inputs and outputs are clear
		if *node.Type == v1.MODEL {
			return nodeInputs, nodeOutputs
		} else {
			fmt.Println("Unkown case.")
			return nil, nil
		}
	} else if *node.Type == v1.MODEL {
		// We ignore all childs except first one
		childInputs, childOutputs := p.GetShapeFromGraph(&node.Children[0])

		// Sanity check if child's input matches the parent output
		if !AssertShapeCompatibility(nodeOutputs, childInputs) {
			fmt.Println("Parent-Child data shape incompatibility!")
			return nil, nil
		}

		return nodeInputs, childOutputs
	} else if *node.Type == v1.COMBINER {
		// Combiner will get as its input a `list` of childs' outputs.
		// Currently we will treat MODEL nodes as having single input and output.
		// This is relevant for the shape compatibility check.
		var prevChildInputs []MetadataTensor
		var combinedChildOutputs = make([]MetadataTensor, len(node.Children))
		for index, child := range node.Children {
			childInputs, childOutputs := p.GetShapeFromGraph(&child)
			// First we check if all child has same kind of input
			if prevChildInputs != nil {
				if !AssertShapeCompatibility(prevChildInputs, childInputs) {
					fmt.Println("Child", child.Name, "has different input than its siblings.")
					return nil, nil
				}
			}
			prevChildInputs = childInputs

			// Combine Child outputs
			if len(childOutputs) > 1 {
				fmt.Println("We expect MODEL nodes to have only one output.")
				return nil, nil
			}
			combinedChildOutputs[index] = childOutputs[0]
		}
		// Now check if combined child outputs matches the combiner's input
		if !AssertShapeCompatibility(nodeInputs, combinedChildOutputs) {
			fmt.Println("Combiner input does not match combined output of childs.")
			return nil, nil
		}
		return prevChildInputs, nodeOutputs
	} else {
		fmt.Println("Unkown case.")
		return nil, nil
	}
}

func AssertShapeCompatibility(
	inputs []MetadataTensor, outputs []MetadataTensor,
) bool {
	if inputs == nil || outputs == nil {
		return false
	}
	for index, lval := range inputs {
		rval := outputs[index]
		if (lval.DataType != rval.DataType) || (len(lval.Shape) != len(rval.Shape)) {
			return false
		}
		for i, lv := range lval.Shape {
			rv := rval.Shape[i]
			if lv != rv {
				return false
			}
		}
	}
	return true
}
