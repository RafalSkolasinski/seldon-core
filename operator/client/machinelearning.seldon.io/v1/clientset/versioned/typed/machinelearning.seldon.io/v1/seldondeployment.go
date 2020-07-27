/*
Copyright 2019 The Seldon Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/
// Code generated by client-gen. DO NOT EDIT.

package v1

import (
	"context"
	"time"

	v1 "github.com/seldonio/seldon-core/operator/apis/machinelearning.seldon.io/v1"
	scheme "github.com/seldonio/seldon-core/operator/client/machinelearning.seldon.io/v1/clientset/versioned/scheme"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	types "k8s.io/apimachinery/pkg/types"
	watch "k8s.io/apimachinery/pkg/watch"
	rest "k8s.io/client-go/rest"
)

// SeldonDeploymentsGetter has a method to return a SeldonDeploymentInterface.
// A group's client should implement this interface.
type SeldonDeploymentsGetter interface {
	SeldonDeployments(namespace string) SeldonDeploymentInterface
}

// SeldonDeploymentInterface has methods to work with SeldonDeployment resources.
type SeldonDeploymentInterface interface {
	Create(*v1.SeldonDeployment) (*v1.SeldonDeployment, error)
	Update(*v1.SeldonDeployment) (*v1.SeldonDeployment, error)
	Delete(name string, options *metav1.DeleteOptions) error
	DeleteCollection(options *metav1.DeleteOptions, listOptions metav1.ListOptions) error
	Get(name string, options metav1.GetOptions) (*v1.SeldonDeployment, error)
	List(opts metav1.ListOptions) (*v1.SeldonDeploymentList, error)
	Watch(opts metav1.ListOptions) (watch.Interface, error)
	Patch(name string, pt types.PatchType, data []byte, subresources ...string) (result *v1.SeldonDeployment, err error)
	SeldonDeploymentExpansion
}

// seldonDeployments implements SeldonDeploymentInterface
type seldonDeployments struct {
	client rest.Interface
	ns     string
}

// newSeldonDeployments returns a SeldonDeployments
func newSeldonDeployments(c *MachinelearningV1Client, namespace string) *seldonDeployments {
	return &seldonDeployments{
		client: c.RESTClient(),
		ns:     namespace,
	}
}

// Get takes name of the seldonDeployment, and returns the corresponding seldonDeployment object, and an error if there is any.
func (c *seldonDeployments) Get(name string, options metav1.GetOptions) (result *v1.SeldonDeployment, err error) {
	result = &v1.SeldonDeployment{}
	err = c.client.Get().
		Namespace(c.ns).
		Resource("seldondeployments").
		Name(name).
		VersionedParams(&options, scheme.ParameterCodec).
		Do(context.Background()).
		Into(result)
	return
}

// List takes label and field selectors, and returns the list of SeldonDeployments that match those selectors.
func (c *seldonDeployments) List(opts metav1.ListOptions) (result *v1.SeldonDeploymentList, err error) {
	var timeout time.Duration
	if opts.TimeoutSeconds != nil {
		timeout = time.Duration(*opts.TimeoutSeconds) * time.Second
	}
	result = &v1.SeldonDeploymentList{}
	err = c.client.Get().
		Namespace(c.ns).
		Resource("seldondeployments").
		VersionedParams(&opts, scheme.ParameterCodec).
		Timeout(timeout).
		Do(context.Background()).
		Into(result)
	return
}

// Watch returns a watch.Interface that watches the requested seldonDeployments.
func (c *seldonDeployments) Watch(opts metav1.ListOptions) (watch.Interface, error) {
	var timeout time.Duration
	if opts.TimeoutSeconds != nil {
		timeout = time.Duration(*opts.TimeoutSeconds) * time.Second
	}
	opts.Watch = true
	return c.client.Get().
		Namespace(c.ns).
		Resource("seldondeployments").
		VersionedParams(&opts, scheme.ParameterCodec).
		Timeout(timeout).
		Watch(context.Background())
}

// Create takes the representation of a seldonDeployment and creates it.  Returns the server's representation of the seldonDeployment, and an error, if there is any.
func (c *seldonDeployments) Create(seldonDeployment *v1.SeldonDeployment) (result *v1.SeldonDeployment, err error) {
	result = &v1.SeldonDeployment{}
	err = c.client.Post().
		Namespace(c.ns).
		Resource("seldondeployments").
		Body(seldonDeployment).
		Do(context.Background()).
		Into(result)
	return
}

// Update takes the representation of a seldonDeployment and updates it. Returns the server's representation of the seldonDeployment, and an error, if there is any.
func (c *seldonDeployments) Update(seldonDeployment *v1.SeldonDeployment) (result *v1.SeldonDeployment, err error) {
	result = &v1.SeldonDeployment{}
	err = c.client.Put().
		Namespace(c.ns).
		Resource("seldondeployments").
		Name(seldonDeployment.Name).
		Body(seldonDeployment).
		Do(context.Background()).
		Into(result)
	return
}

// Delete takes name of the seldonDeployment and deletes it. Returns an error if one occurs.
func (c *seldonDeployments) Delete(name string, options *metav1.DeleteOptions) error {
	return c.client.Delete().
		Namespace(c.ns).
		Resource("seldondeployments").
		Name(name).
		Body(options).
		Do(context.Background()).
		Error()
}

// DeleteCollection deletes a collection of objects.
func (c *seldonDeployments) DeleteCollection(options *metav1.DeleteOptions, listOptions metav1.ListOptions) error {
	var timeout time.Duration
	if listOptions.TimeoutSeconds != nil {
		timeout = time.Duration(*listOptions.TimeoutSeconds) * time.Second
	}
	return c.client.Delete().
		Namespace(c.ns).
		Resource("seldondeployments").
		VersionedParams(&listOptions, scheme.ParameterCodec).
		Timeout(timeout).
		Body(options).
		Do(context.Background()).
		Error()
}

// Patch applies the patch and returns the patched seldonDeployment.
func (c *seldonDeployments) Patch(name string, pt types.PatchType, data []byte, subresources ...string) (result *v1.SeldonDeployment, err error) {
	result = &v1.SeldonDeployment{}
	err = c.client.Patch(pt).
		Namespace(c.ns).
		Resource("seldondeployments").
		SubResource(subresources...).
		Name(name).
		Body(data).
		Do(context.Background()).
		Into(result)
	return
}
