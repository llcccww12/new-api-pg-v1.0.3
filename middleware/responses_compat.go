package middleware

import (
	"bytes"
	"encoding/json"
	"io"

	"github.com/gin-gonic/gin"
)

// ResponsesCompatToChatCompletions normalizes an OpenAI Responses API style
// request body into the OpenAI Chat Completions shape so that it can be
// served by RelayFormatOpenAI code paths (which only know about
// "messages" and nested "tools").
//
// Cursor IDE Agent mode sends Responses API payloads (input + flat tools)
// to /v1/chat/completions endpoints when "Override OpenAI Base URL" is
// configured to a custom OpenAI-compatible provider. Without this shim,
// the relay handler rejects the request with "field messages is required".
//
// Conversions:
//   - input (Responses)  -> messages (Chat Completions)
//   - flat tools [{type, name, description, parameters}] -> nested
//     [{type: "function", function: {name, description, parameters}}]
//   - non-standard tool types ("custom", "apply_patch", ...) -> filtered out
//   - reasoning: {effort: "..."} -> reasoning_effort: "..."
//   - input_text/output_text items -> text role content strings
//
// The middleware is a no-op for request bodies that already have a
// "messages" field, so existing Chat Completions clients are unaffected.
func ResponsesCompatToChatCompletions() gin.HandlerFunc {
	return func(c *gin.Context) {
		body, err := io.ReadAll(c.Request.Body)
		if err != nil {
			c.Next()
			return
		}
		defer func() { c.Request.Body = io.NopCloser(bytes.NewReader(body)) }()

		var probe map[string]json.RawMessage
		if err := json.Unmarshal(body, &probe); err != nil {
			c.Next()
			return
		}
		if _, hasMessages := probe["messages"]; hasMessages {
			c.Next()
			return
		}
		if _, hasInput := probe["input"]; !hasInput {
			c.Next()
			return
		}

		converted, ok := convertResponsesToChatCompletions(body)
		if !ok {
			c.Next()
			return
		}
		c.Request.Body = io.NopCloser(bytes.NewReader(converted))
		c.Request.ContentLength = int64(len(converted))
c.Next()
	}
}

func convertResponsesToChatCompletions(body []byte) ([]byte, bool) {
	var src map[string]json.RawMessage
	if err := json.Unmarshal(body, &src); err != nil {
		return nil, false
	}
	out := map[string]json.RawMessage{}

	if v, ok := src["model"]; ok {
		out["model"] = v
	}
	if v, ok := src["input"]; ok {
		msgs, ok := convertResponsesInput(v)
		if !ok {
			return nil, false
		}
		out["messages"] = msgs
	}
	if v, ok := src["stream"]; ok {
		out["stream"] = v
	}
	if v, ok := src["max_completion_tokens"]; ok {
		out["max_completion_tokens"] = v
	} else if v, ok := src["max_output_tokens"]; ok {
		out["max_completion_tokens"] = v
	}
	for _, k := range []string{"temperature", "top_p", "top_k", "stop", "user"} {
		if v, ok := src[k]; ok {
			out[k] = v
		}
	}
	if r, ok := src["reasoning"]; ok {
		var rj struct {
			Effort string `json:"effort"`
		}
		if err := json.Unmarshal(r, &rj); err == nil && rj.Effort != "" {
			b, _ := json.Marshal(rj.Effort)
			out["reasoning_effort"] = b
		}
	}
	if v, ok := src["tools"]; ok {
		if t, ok := convertResponsesTools(v); ok {
			out["tools"] = t
		}
	}
	if v, ok := src["tool_choice"]; ok {
		out["tool_choice"] = v
	}
	if v, ok := src["text"]; ok {
		var tj struct {
			Format json.RawMessage `json:"format"`
		}
		if err := json.Unmarshal(v, &tj); err == nil && len(tj.Format) > 0 {
			out["response_format"] = tj.Format
		}
	}
	b, err := json.Marshal(out)
	if err != nil {
		return nil, false
	}
	return b, true
}

func convertResponsesInput(input json.RawMessage) (json.RawMessage, bool) {
	var arr []json.RawMessage
	if err := json.Unmarshal(input, &arr); err != nil {
		return nil, false
	}
	messages := make([]map[string]any, 0, len(arr))
	for _, raw := range arr {
		var item struct {
			Type       string          `json:"type"`
			Role       string          `json:"role"`
			Content    json.RawMessage `json:"content"`
			Output     json.RawMessage `json:"output"`
			CallID     string          `json:"call_id"`
			Name       string          `json:"name"`
			Arguments  json.RawMessage `json:"arguments"`
			Message    json.RawMessage `json:"message"`
			ToolCallID string          `json:"tool_call_id"`
		}
		if err := json.Unmarshal(raw, &item); err != nil {
			continue
		}
		switch item.Type {
		case "function_call_output":
			outputStr := string(item.Output)
			if len(outputStr) == 0 {
				outputStr = "{}"
			}
			messages = append(messages, map[string]any{
				"role":         "tool",
				"tool_call_id": item.CallID,
				"content":      outputStr,
			})
		case "function_call":
			tc := map[string]any{
				"id":       item.CallID,
				"type":     "function",
				"function": map[string]any{"name": item.Name, "arguments": string(item.Arguments)},
			}
			messages = append(messages, map[string]any{
				"role":       "assistant",
				"content":    "",
				"tool_calls": []any{tc},
			})
		case "message":
			contentStr := extractResponsesContent(item.Content)
			messages = append(messages, map[string]any{
				"role":    "assistant",
				"content": contentStr,
			})
		default:
			role := item.Role
			if role == "" {
				role = "user"
			}
			contentStr := extractResponsesContent(item.Content)
			msg := map[string]any{"role": role, "content": contentStr}
			if item.ToolCallID != "" {
				msg["tool_call_id"] = item.ToolCallID
			}
			messages = append(messages, msg)
		}
	}
	b, err := json.Marshal(messages)
	if err != nil {
		return nil, false
	}
	return b, true
}

func extractResponsesContent(content json.RawMessage) string {
	if len(content) == 0 {
		return ""
	}
	var s string
	if err := json.Unmarshal(content, &s); err == nil {
		return s
	}
	var arr []map[string]any
	if err := json.Unmarshal(content, &arr); err == nil {
		var out string
		for _, p := range arr {
			t, _ := p["type"].(string)
			if t == "output_text" || t == "input_text" || t == "text" {
				if txt, _ := p["text"].(string); txt != "" {
					if out != "" {
						out += "\n"
					}
					out += txt
				}
			}
		}
		return out
	}
	return string(content)
}

func convertResponsesTools(tools json.RawMessage) (json.RawMessage, bool) {
	var arr []map[string]any
	if err := json.Unmarshal(tools, &arr); err != nil {
		return nil, false
	}
	out := make([]map[string]any, 0, len(arr))
	for _, t := range arr {
		typ, _ := t["type"].(string)
		if typ != "function" {
			continue
		}
		name, _ := t["name"].(string)
		if name == "" {
			continue
		}
		fn := map[string]any{"name": name}
		if d, ok := t["description"]; ok {
			fn["description"] = d
		}
		if p, ok := t["parameters"]; ok {
			fn["parameters"] = p
		}
		if s, ok := t["strict"]; ok {
			fn["strict"] = s
		}
		out = append(out, map[string]any{
			"type":     "function",
			"function": fn,
		})
	}
	if len(out) == 0 {
		return []byte("[]"), true
	}
	b, err := json.Marshal(out)
	if err != nil {
		return nil, false
	}
	return b, true
}
