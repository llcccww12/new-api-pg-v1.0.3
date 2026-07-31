package aws

import (
	"context"
	"encoding/json"
	"io"
	"net/http"

	"github.com/QuantumNous/new-api/common"
	"github.com/QuantumNous/new-api/dto"
	"github.com/QuantumNous/new-api/logger"
)

type AwsClaudeRequest struct {
	AnthropicVersion string              `json:"anthropic_version"`
	AnthropicBeta    json.RawMessage     `json:"anthropic_beta,omitempty"`
	System           any                 `json:"system,omitempty"`
	Messages         []dto.ClaudeMessage `json:"messages"`
	MaxTokens        uint                `json:"max_tokens,omitempty"`
	Temperature      *float64            `json:"temperature,omitempty"`
	TopP             float64             `json:"top_p,omitempty"`
	TopK             int                 `json:"top_k,omitempty"`
	StopSequences    []string            `json:"stop_sequences,omitempty"`
	Tools            any                 `json:"tools,omitempty"`
	ToolChoice       any                 `json:"tool_choice,omitempty"`
	Thinking         *dto.Thinking       `json:"thinking,omitempty"`
	OutputConfig     json.RawMessage     `json:"output_config,omitempty"`
}

func removeCacheControl(content any) any {
	switch v := content.(type) {
	case []interface{}:
		for _, item := range v {
			if m, ok := item.(map[string]interface{}); ok {
				delete(m, "cache_control")
			}
		}
	case []map[string]interface{}:
		for _, m := range v {
			delete(m, "cache_control")
		}
	}
	return content
}

func formatRequest(requestBody io.Reader, requestHeader http.Header) (*AwsClaudeRequest, error) {
	var awsClaudeRequest AwsClaudeRequest
	err := common.DecodeJson(requestBody, &awsClaudeRequest)
	if err != nil {
		return nil, err
	}
	awsClaudeRequest.AnthropicVersion = "bedrock-2023-05-31"

	// Bedrock 要求 max_tokens 必填：OpenAI 格式(/v1/chat/completions)请求里该字段可选，
	// 转换后可能为 0 被 omitempty 丢弃，这里兜底补默认值，避免 400 "max_tokens: Field required"
	if awsClaudeRequest.MaxTokens == 0 {
		awsClaudeRequest.MaxTokens = 4096
	}

	for i := range awsClaudeRequest.Messages {
		awsClaudeRequest.Messages[i].Content = removeCacheControl(awsClaudeRequest.Messages[i].Content)
	}
	if awsClaudeRequest.System != nil {
		awsClaudeRequest.System = removeCacheControl(awsClaudeRequest.System)
	}
	awsClaudeRequest.AnthropicBeta = nil
	logger.LogJson(context.Background(), "json", awsClaudeRequest)
	return &awsClaudeRequest, nil
}

type NovaMessage struct {
	Role    string        `json:"role"`
	Content []NovaContent `json:"content"`
}

type NovaContent struct {
	Text string `json:"text"`
}

type NovaRequest struct {
	SchemaVersion   string               `json:"schemaVersion"`
	Messages        []NovaMessage        `json:"messages"`
	InferenceConfig *NovaInferenceConfig `json:"inferenceConfig,omitempty"`
}

type NovaInferenceConfig struct {
	MaxTokens     int      `json:"maxTokens,omitempty"`
	Temperature   float64  `json:"temperature,omitempty"`
	TopP          float64  `json:"topP,omitempty"`
	TopK          int      `json:"topK,omitempty"`
	StopSequences []string `json:"stopSequences,omitempty"`
}

func convertToNovaRequest(req *dto.GeneralOpenAIRequest) *NovaRequest {
	novaMessages := make([]NovaMessage, len(req.Messages))
	for i, msg := range req.Messages {
		novaMessages[i] = NovaMessage{
			Role:    msg.Role,
			Content: []NovaContent{{Text: msg.StringContent()}},
		}
	}
	novaReq := &NovaRequest{
		SchemaVersion: "messages-v1",
		Messages:      novaMessages,
	}
	if (req.MaxTokens != nil && *req.MaxTokens != 0) || (req.Temperature != nil && *req.Temperature != 0) || (req.TopP != nil && *req.TopP != 0) || (req.TopK != nil && *req.TopK != 0) || req.Stop != nil {
		novaReq.InferenceConfig = &NovaInferenceConfig{}
		if req.MaxTokens != nil && *req.MaxTokens != 0 {
			novaReq.InferenceConfig.MaxTokens = int(*req.MaxTokens)
		}
		if req.Temperature != nil && *req.Temperature != 0 {
			novaReq.InferenceConfig.Temperature = *req.Temperature
		}
		if req.TopP != nil && *req.TopP != 0 {
			novaReq.InferenceConfig.TopP = *req.TopP
		}
		if req.TopK != nil && *req.TopK != 0 {
			novaReq.InferenceConfig.TopK = *req.TopK
		}
		if req.Stop != nil {
			if stopSequences := parseStopSequences(req.Stop); len(stopSequences) > 0 {
				novaReq.InferenceConfig.StopSequences = stopSequences
			}
		}
	}
	return novaReq
}

func parseStopSequences(stop any) []string {
	if stop == nil {
		return nil
	}
	switch v := stop.(type) {
	case string:
		if v != "" {
			return []string{v}
		}
	case []string:
		return v
	case []interface{}:
		var sequences []string
		for _, item := range v {
			if str, ok := item.(string); ok && str != "" {
				sequences = append(sequences, str)
			}
		}
		return sequences
	}
	return nil
}
