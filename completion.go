package openai

import (
	"context"
	"errors"
	"net/http"
)

var (
	ErrCompletionUnsupportedModel              = errors.New("this model is not supported with this method, please use CreateChatCompletion client method instead") //nolint:lll
	ErrCompletionStreamNotSupported            = errors.New("streaming is not supported with this method, please use CreateCompletionStream")                      //nolint:lll
	ErrCompletionRequestPromptTypeNotSupported = errors.New("the type of CompletionRequest.Prompt only supports string and []string")                              //nolint:lll
)

// GPT3 Defines the models provided by OpenAI to use when generating
// completions from OpenAI.
// GPT3 Models are designed for text-based tasks. For code-specific
// tasks, please refer to the Codex series of models.
const (
	GPT432K0613           = "gpt-4-32k-0613"
	GPT432K0314           = "gpt-4-32k-0314"
	GPT432K               = "gpt-4-32k"
	GPT40613              = "gpt-4-0613"
	GPT40314              = "gpt-4-0314"
	GPT4TurboPreview      = "gpt-4-1106-preview"
	GPT4VisionPreview     = "gpt-4-vision-preview"
	GPT4                  = "gpt-4"
	GPT3Dot5Turbo1106     = "gpt-3.5-turbo-1106"
	GPT3Dot5Turbo0613     = "gpt-3.5-turbo-0613"
	GPT3Dot5Turbo0301     = "gpt-3.5-turbo-0301"
	GPT3Dot5Turbo16K      = "gpt-3.5-turbo-16k"
	GPT3Dot5Turbo16K0613  = "gpt-3.5-turbo-16k-0613"
	GPT3Dot5Turbo         = "gpt-3.5-turbo"
	GPT3Dot5TurboInstruct = "gpt-3.5-turbo-instruct"
	// Deprecated: Will be shut down on January 04, 2024. Use gpt-3.5-turbo-instruct instead.
	GPT3TextDavinci003 = "text-davinci-003"
	// Deprecated: Will be shut down on January 04, 2024. Use gpt-3.5-turbo-instruct instead.
	GPT3TextDavinci002 = "text-davinci-002"
	// Deprecated: Will be shut down on January 04, 2024. Use gpt-3.5-turbo-instruct instead.
	GPT3TextCurie001 = "text-curie-001"
	// Deprecated: Will be shut down on January 04, 2024. Use gpt-3.5-turbo-instruct instead.
	GPT3TextBabbage001 = "text-babbage-001"
	// Deprecated: Will be shut down on January 04, 2024. Use gpt-3.5-turbo-instruct instead.
	GPT3TextAda001 = "text-ada-001"
	// Deprecated: Will be shut down on January 04, 2024. Use gpt-3.5-turbo-instruct instead.
	GPT3TextDavinci001 = "text-davinci-001"
	// Deprecated: Will be shut down on January 04, 2024. Use gpt-3.5-turbo-instruct instead.
	GPT3DavinciInstructBeta = "davinci-instruct-beta"
	GPT3Davinci             = "davinci"
	GPT3Davinci002          = "davinci-002"
	// Deprecated: Will be shut down on January 04, 2024. Use gpt-3.5-turbo-instruct instead.
	GPT3CurieInstructBeta = "curie-instruct-beta"
	GPT3Curie             = "curie"
	GPT3Curie002          = "curie-002"
	GPT3Ada               = "ada"
	GPT3Ada002            = "ada-002"
	GPT3Babbage           = "babbage"
	GPT3Babbage002        = "babbage-002"
)

// Codex Defines the models provided by OpenAI.
// These models are designed for code-specific tasks, and use
// a different tokenizer which optimizes for whitespace.
const (
	CodexCodeDavinci002 = "code-davinci-002"
	CodexCodeCushman001 = "code-cushman-001"
	CodexCodeDavinci001 = "code-davinci-001"
)

var disabledModelsForEndpoints = map[string]map[string]bool{
	"/completions": {
		GPT3Dot5Turbo:        true,
		GPT3Dot5Turbo0301:    true,
		GPT3Dot5Turbo0613:    true,
		GPT3Dot5Turbo1106:    true,
		GPT3Dot5Turbo16K:     true,
		GPT3Dot5Turbo16K0613: true,
		GPT4:                 true,
		GPT4TurboPreview:     true,
		GPT4VisionPreview:    true,
		GPT40314:             true,
		GPT40613:             true,
		GPT432K:              true,
		GPT432K0314:          true,
		GPT432K0613:          true,
	},
	chatCompletionsSuffix: {
		CodexCodeDavinci002:     true,
		CodexCodeCushman001:     true,
		CodexCodeDavinci001:     true,
		GPT3TextDavinci003:      true,
		GPT3TextDavinci002:      true,
		GPT3TextCurie001:        true,
		GPT3TextBabbage001:      true,
		GPT3TextAda001:          true,
		GPT3TextDavinci001:      true,
		GPT3DavinciInstructBeta: true,
		GPT3Davinci:             true,
		GPT3CurieInstructBeta:   true,
		GPT3Curie:               true,
		GPT3Ada:                 true,
		GPT3Babbage:             true,
	},
}

func checkEndpointSupportsModel(endpoint, model string) bool {
	return !disabledModelsForEndpoints[endpoint][model]
}

func checkPromptType(prompt any) bool {
	_, isString := prompt.(string)
	_, isStringSlice := prompt.([]string)
	return isString || isStringSlice
}

// CompletionRequest represents a request structure for completion API.
type CompletionRequest struct {
	Model            string   `json:"model"`
	Prompt           any      `json:"prompt,omitempty"`
	Suffix           string   `json:"suffix,omitempty"`
	MaxTokens        int      `json:"max_tokens,omitempty"`
	Temperature      float32  `json:"temperature,omitempty"`
	TopP             float32  `json:"top_p,omitempty"`
	N                int      `json:"n,omitempty"`
	Stream           bool     `json:"stream,omitempty"`
	LogProbs         int      `json:"logprobs,omitempty"`
	Echo             bool     `json:"echo,omitempty"`
	Stop             []string `json:"stop,omitempty"`
	PresencePenalty  float32  `json:"presence_penalty,omitempty"`
	FrequencyPenalty float32  `json:"frequency_penalty,omitempty"`
	BestOf           int      `json:"best_of,omitempty"`
	// LogitBias is must be a token id string (specified by their token ID in the tokenizer), not a word string.
	// incorrect: `"logit_bias":{"You": 6}`, correct: `"logit_bias":{"1639": 6}`
	// refs: https://platform.openai.com/docs/api-reference/completions/create#completions/create-logit_bias
	LogitBias map[string]int `json:"logit_bias,omitempty"`
	User      string         `json:"user,omitempty"`
}

type CompletionCustomRequest struct {
	Model                    string   `json:"model"`
	Prompt                   string   `json:"prompt,omitempty"`
	BestOf                   int      `json:"best_of,omitempty"`
	Echo                     bool     `json:"echo,omitempty"`
	FrequencyPenalty         float32  `json:"frequency_penalty,omitempty"`
	Logprobs                 int      `json:"logprobs,omitempty"`
	MaxTokens                int      `json:"max_tokens,omitempty"`
	N                        int      `json:"n,omitempty"`
	PresencePenalty          float32  `json:"presence_penalty,omitempty"`
	Stop                     []string `json:"stop,omitempty"`
	Stream                   bool     `json:"stream,omitempty"`
	Suffix                   string   `json:"suffix,omitempty"`
	Temperature              float32  `json:"temperature,omitempty"`
	TopP                     float32  `json:"top_p,omitempty"`
	User                     string   `json:"user,omitempty"`
	Preset                   string   `json:"preset,omitempty"`
	MinP                     int      `json:"min_p,omitempty"`
	TopK                     int      `json:"top_k,omitempty"`
	RepetitionPenalty        float32  `json:"repetition_penalty,omitempty"`
	RepetitionPenaltyRange   int      `json:"repetition_penalty_range,omitempty"`
	TypicalP                 int      `json:"typical_p,omitempty"`
	Tfs                      int      `json:"tfs,omitempty"`
	TopA                     int      `json:"top_a,omitempty"`
	EpsilonCutoff            int      `json:"epsilon_cutoff,omitempty"`
	EtaCutoff                int      `json:"eta_cutoff,omitempty"`
	GuidanceScale            int      `json:"guidance_scale,omitempty"`
	NegativePrompt           string   `json:"negative_prompt,omitempty"`
	PenaltyAlpha             int      `json:"penalty_alpha,omitempty"`
	MirostatMode             int      `json:"mirostat_mode,omitempty"`
	MirostatTau              int      `json:"mirostat_tau,omitempty"`
	MirostatEta              float32  `json:"mirostat_eta,omitempty"`
	TemperatureLast          bool     `json:"temperature_last,omitempty"`
	DoSample                 bool     `json:"do_sample,omitempty"`
	Seed                     int      `json:"seed,omitempty"`
	LogitBias 		map[string]int	  `json:"logit_bias,omitempty"`
	EncoderRepetitionPenalty int      `json:"encoder_repetition_penalty,omitempty"`
	NoRepeatNgramSize        int      `json:"no_repeat_ngram_size,omitempty"`
	MinLength                int      `json:"min_length,omitempty"`
	NumBeams                 int      `json:"num_beams,omitempty"`
	LengthPenalty            int      `json:"length_penalty,omitempty"`
	EarlyStopping            bool     `json:"early_stopping,omitempty"`
	TruncationLength         int      `json:"truncation_length,omitempty"`
	MaxTokensSecond          int      `json:"max_tokens_second,omitempty"`
	CustomTokenBans          string   `json:"custom_token_bans,omitempty"`
	AutoMaxNewTokens         bool     `json:"auto_max_new_tokens,omitempty"`
	BanEosToken              bool     `json:"ban_eos_token,omitempty"`
	AddBosToken              bool     `json:"add_bos_token,omitempty"`
	SkipSpecialTokens        bool     `json:"skip_special_tokens,omitempty"`
	GrammarString            string   `json:"grammar_string,omitempty"`
}

// CompletionChoice represents one of possible completions.
type CompletionChoice struct {
	Text         string        `json:"text"`
	Index        int           `json:"index"`
	FinishReason string        `json:"finish_reason"`
	LogProbs     LogprobResult `json:"logprobs"`
}

// LogprobResult represents logprob result of Choice.
type LogprobResult struct {
	Tokens        []string             `json:"tokens"`
	TokenLogprobs []float32            `json:"token_logprobs"`
	TopLogprobs   []map[string]float32 `json:"top_logprobs"`
	TextOffset    []int                `json:"text_offset"`
}

// CompletionResponse represents a response structure for completion API.
type CompletionResponse struct {
	ID      string             `json:"id"`
	Object  string             `json:"object"`
	Created int64              `json:"created"`
	Model   string             `json:"model"`
	Choices []CompletionChoice `json:"choices"`
	Usage   Usage              `json:"usage"`

	httpHeader
}

// CreateCompletion — API call to create a completion. This is the main endpoint of the API. Returns new text as well
// as, if requested, the probabilities over each alternative token at each position.
//
// If using a fine-tuned model, simply provide the model's ID in the CompletionRequest object,
// and the server will use the model's parameters to generate the completion.
func (c *Client) CreateCompletion(
	ctx context.Context,
	request CompletionRequest,
) (response CompletionResponse, err error) {
	if request.Stream {
		err = ErrCompletionStreamNotSupported
		return
	}

	urlSuffix := "/completions"
	if !checkEndpointSupportsModel(urlSuffix, request.Model) {
		err = ErrCompletionUnsupportedModel
		return
	}

	if !checkPromptType(request.Prompt) {
		err = ErrCompletionRequestPromptTypeNotSupported
		return
	}

	req, err := c.newRequest(ctx, http.MethodPost, c.fullURL(urlSuffix, request.Model), withBody(request))
	if err != nil {
		return
	}

	err = c.sendRequest(req, &response)
	return
}

func (c *Client) CreateCompletionByCustom(
	ctx context.Context,
	request CompletionCustomRequest,
) (response CompletionResponse, err error) {
	if request.Stream {
		err = ErrCompletionStreamNotSupported
		return
	}

	urlSuffix := "/completions"
	if !checkEndpointSupportsModel(urlSuffix, request.Model) {
		err = ErrCompletionUnsupportedModel
		return
	}

	if !checkPromptType(request.Prompt) {
		err = ErrCompletionRequestPromptTypeNotSupported
		return
	}

	req, err := c.newRequest(ctx, http.MethodPost, c.fullURL(urlSuffix, request.Model), withBody(request))
	if err != nil {
		return
	}

	err = c.sendRequest(req, &response)
	return
}