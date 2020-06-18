package seldon

import (
	"context"
	"net/url"

	"github.com/go-logr/logr"
	"github.com/seldonio/seldon-core/executor/api/client"
	"github.com/seldonio/seldon-core/executor/api/grpc"
	"github.com/seldonio/seldon-core/executor/api/grpc/seldon/proto"
	"github.com/seldonio/seldon-core/executor/api/payload"
	"github.com/seldonio/seldon-core/executor/predictor"
	v1 "github.com/seldonio/seldon-core/operator/apis/machinelearning.seldon.io/v1"
	// codes "google.golang.org/grpc/codes"
	// status "google.golang.org/grpc/status"
	"fmt"
	logf "sigs.k8s.io/controller-runtime/pkg/runtime/log"
)

type GrpcSeldonServer struct {
	Client    client.SeldonApiClient
	predictor *v1.PredictorSpec
	Log       logr.Logger
	ServerUrl *url.URL
	Namespace string
}

func NewGrpcSeldonServer(predictor *v1.PredictorSpec, client client.SeldonApiClient, serverUrl *url.URL, namespace string) *GrpcSeldonServer {
	return &GrpcSeldonServer{
		Client:    client,
		predictor: predictor,
		Log:       logf.Log.WithName("SeldonGrpcApi"),
		ServerUrl: serverUrl,
		Namespace: namespace,
	}
}

func (g GrpcSeldonServer) Predict(ctx context.Context, req *proto.SeldonMessage) (*proto.SeldonMessage, error) {
	md := grpc.CollectMetadata(ctx)
	ctx = context.WithValue(ctx, payload.SeldonPUIDHeader, md.Get(payload.SeldonPUIDHeader)[0])
	seldonPredictorProcess := predictor.NewPredictorProcess(ctx, g.Client, logf.Log.WithName("SeldonMessageRestClient"), g.ServerUrl, g.Namespace, md)
	reqPayload := payload.ProtoPayload{Msg: req}
	resPayload, err := seldonPredictorProcess.Predict(g.predictor.Graph, &reqPayload)
	if err != nil {
		g.Log.Error(err, "Failed to call predict")
		return payloadToMessage(resPayload), err
	}
	return payloadToMessage(resPayload), nil
}

func (g GrpcSeldonServer) SendFeedback(ctx context.Context, req *proto.Feedback) (*proto.SeldonMessage, error) {
	seldonPredictorProcess := predictor.NewPredictorProcess(ctx, g.Client, logf.Log.WithName("SeldonMessageRestClient"), g.ServerUrl, g.Namespace, grpc.CollectMetadata(ctx))
	reqPayload := payload.ProtoPayload{Msg: req}
	resPayload, err := seldonPredictorProcess.Feedback(g.predictor.Graph, &reqPayload)
	if err != nil {
		g.Log.Error(err, "Failed to call feedback")
		return payloadToMessage(resPayload), err
	}
	return payloadToMessage(resPayload), nil
}

func (g GrpcSeldonServer) Metadata(ctx context.Context, req *proto.SeldonModelMetadataRequest) (*proto.SeldonModelMetadata, error) {
	g.Log.Info("I have been called!")
	seldonPredictorProcess := predictor.NewPredictorProcess(ctx, g.Client, logf.Log.WithName("SeldonMessageRestClient"), g.ServerUrl, g.Namespace, grpc.CollectMetadata(ctx))
	modelName := req.GetName()
	fmt.Println("modelName:", modelName)

	resPayload, err := seldonPredictorProcess.Metadata(g.predictor.Graph, modelName, nil)
	fmt.Println(resPayload, err)
	if err != nil {
		return nil, err
	}

	return payloadToModelMetadata(resPayload), nil
}

func modelMetadata(p *predictor.PredictorProcess, name string) {

}

func payloadToMessage(p payload.SeldonPayload) *proto.SeldonMessage {
	if m, ok := p.GetPayload().(*proto.SeldonMessage); ok {
		return m
	}
	return nil
}

func payloadToModelMetadata(p payload.SeldonPayload) *proto.SeldonModelMetadata {
	if m, ok := p.GetPayload().(*proto.SeldonModelMetadata); ok {
		return m
	}
	return nil
}
