package predictor

import (
	"fmt"
	"github.com/seldonio/seldon-core/operator/apis/machinelearning.seldon.io/v1"
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
	Name         string                   `json:"name,omitempty"`
	Models       map[string]ModelMetadata `json:"models,omitempty"`
	GraphInputs  []MetadataTensor         `json:"graphinputs,omitempty"`
	GraphOutputs []MetadataTensor         `json:"graphoutputs,omitempty"`
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

	// Node has no children
	if node.Children == nil || len(node.Children) == 0 {
		// If Node is model then inputs and outputs are clear
		if *node.Type == v1.MODEL {
			return nodeInputs, nodeOutputs
		} else {
			fmt.Println("Unkown case.")
			return nil, nil
		}
	} else if (*node.Type == v1.MODEL || *node.Type == v1.TRANSFORMER)  {
		// We ignore all childs except first one
		childInputs, childOutputs := p.GetShapeFromGraph(&node.Children[0])

		// Sanity check if child's input matches the parent output
		if !AssertShapeCompatibility(nodeOutputs, childInputs) {
			fmt.Println(nodeOutputs, childInputs)
			return nil, nil
		}

		return nodeInputs, childOutputs
	} else if *node.Type == v1.OUTPUT_TRANSFORMER {
		// OUTPUT_TRANSFORMER passes its input to childs and then processes the output
		// We ignore all childs except first one
		childInputs, childOutputs := p.GetShapeFromGraph(&node.Children[0])

		// Sanity check if child's output matches the parent input
		if !AssertShapeCompatibility(nodeInputs, childOutputs) {
			fmt.Println(nodeOutputs, childOutputs)
			return nil, nil
		}

		return childInputs, nodeOutputs
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
	} else if *node.Type == v1.ROUTER {
		// ROUTER will request to one of its childs and return child's output.
		// We only check if all children have same inputs and outputs and set
		// single input/output as node input / output.
		var prevChildInputs []MetadataTensor
		var prevChildOutputs []MetadataTensor
		for _, child := range node.Children {
			childInputs, childOutputs := p.GetShapeFromGraph(&child)
			// First we check if all child has same kind of input and output
			if prevChildInputs != nil {
				if !AssertShapeCompatibility(prevChildInputs, childInputs) {
					fmt.Println("Child", child.Name, "has different input than its siblings.")
					return nil, nil
				}
			}
			if prevChildOutputs != nil {
				if !AssertShapeCompatibility(prevChildOutputs, childOutputs) {
					fmt.Println("Child", child.Name, "has different input than its siblings.")
					return nil, nil
				}
			}
			prevChildInputs = childInputs
			prevChildOutputs = childOutputs
		}

		return prevChildInputs, prevChildOutputs
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
