// Copyright © 2026 Apple Inc.

import AVFoundation
import CoreImage
import Foundation
import MLX
import MLXLMCommon
import MLXNN

// MARK: - Configuration

/// Configuration of `google/embeddinggemma-2` (`model_type: embedding_gemma2`).
///
/// Decodes the checkpoint's `config.json` directly, including the nested `text_config`
/// and the optional `vision_config` and `audio_config`.
public struct EmbeddingGemma2Configuration: Codable, Sendable {

    /// Context window from the model card. `max_position_embeddings` is the RoPE table
    /// size, not a usable limit.
    public static let contextLength = 8_192

    /// One attention head layout per layer: global layers use larger heads and one key/value head.
    public struct Attention: Equatable, Sendable {
        public let headDim: Int
        public let kvHeads: Int
        public let ropeBase: Float
        public let isGlobal: Bool
    }

    /// The vision encoder layout and the tokens that mark images and video frames in a
    /// sequence. Video frames share the image begin and end markers.
    public struct Vision: Sendable {
        public let encoder: Gemma4VisionConfiguration
        public let imageTokenID: Int
        public let videoTokenID: Int?
        public let beginImageTokenID: Int
        public let endImageTokenID: Int
    }

    /// The audio encoder layout and the tokens that mark audio in a sequence.
    public struct Audio: Sendable {
        public let encoder: Gemma4AudioConfiguration
        public let audioTokenID: Int
        public let beginAudioTokenID: Int
        public let endAudioTokenID: Int
    }

    public let hiddenSize: Int
    public let hiddenLayers: Int
    public let attentionHeads: Int
    public let kvHeads: Int
    public let headDim: Int
    public let intermediateSize: Int
    public let perLayerInputSize: Int
    public let embeddingDim: Int
    public let vocabularySize: Int
    public let rmsNormEps: Float
    /// Radius: a token sees the keys within this distance on both sides.
    public let slidingWindow: Int
    public let layerTypes: [String]
    public let attention: [Attention]
    public private(set) var vision: Vision?
    public private(set) var audio: Audio?

    /// The image, video and audio placeholders that soft tokens replace.
    public var softTokenIDs: [Int] {
        [vision?.imageTokenID, vision?.videoTokenID, audio?.audioTokenID].compactMap { $0 }
    }

    private struct LayerOverride: Codable {
        let headDim: Int?
        let keyValueHeads: Int?

        enum CodingKeys: String, CodingKey {
            case headDim = "head_dim"
            case keyValueHeads = "num_key_value_heads"
        }
    }

    private struct RopeParameters: Codable {
        let ropeTheta: Float
        let ropeType: String?

        enum CodingKeys: String, CodingKey {
            case ropeTheta = "rope_theta"
            case ropeType = "rope_type"
        }
    }

    /// Text fields. A multimodal checkpoint nests them under `text_config`.
    private struct TextConfig: Codable {
        let hiddenSize: Int
        let hiddenLayers: Int
        let attentionHeads: Int
        let keyValueHeads: Int?
        let headDim: Int
        let intermediateSize: Int
        let perLayerInputSize: Int
        let embeddingDim: Int
        let vocabularySize: Int
        let rmsNormEps: Float
        let slidingWindow: Int
        let layerTypes: [String]
        let perLayerConfig: [String: LayerOverride]?
        let ropeParameters: [String: RopeParameters]

        enum CodingKeys: String, CodingKey {
            case hiddenSize = "hidden_size"
            case hiddenLayers = "num_hidden_layers"
            case attentionHeads = "num_attention_heads"
            case keyValueHeads = "num_key_value_heads"
            case headDim = "head_dim"
            case intermediateSize = "intermediate_size"
            case perLayerInputSize = "hidden_size_per_layer_input"
            case embeddingDim = "embedding_dim"
            case vocabularySize = "vocab_size"
            case rmsNormEps = "rms_norm_eps"
            case slidingWindow = "sliding_window"
            case layerTypes = "layer_types"
            case perLayerConfig = "per_layer_config"
            case ropeParameters = "rope_parameters"
        }
    }

    private enum RootKeys: String, CodingKey {
        case textConfig = "text_config"
        case visionConfig = "vision_config"
        case audioConfig = "audio_config"
        case imageTokenId = "image_token_id"
        case videoTokenId = "video_token_id"
        case boiTokenId = "boi_token_id"
        case eoiTokenId = "eoi_token_id"
        case audioTokenId = "audio_token_id"
        case boaTokenId = "boa_token_id"
        case eoaTokenId = "eoa_token_id"
        case eoaTokenIndex = "eoa_token_index"
    }

    public init(from decoder: any Decoder) throws {
        let root = try decoder.container(keyedBy: RootKeys.self)
        // Standalone text checkpoints keep the same fields at the top level.
        let text =
            try root.decodeIfPresent(TextConfig.self, forKey: .textConfig)
            ?? TextConfig(from: decoder)

        hiddenSize = text.hiddenSize
        hiddenLayers = text.hiddenLayers
        attentionHeads = text.attentionHeads
        kvHeads = text.keyValueHeads ?? text.attentionHeads
        headDim = text.headDim
        intermediateSize = text.intermediateSize
        perLayerInputSize = text.perLayerInputSize
        embeddingDim = text.embeddingDim
        vocabularySize = text.vocabularySize
        rmsNormEps = text.rmsNormEps
        slidingWindow = text.slidingWindow
        layerTypes = text.layerTypes

        let overrides = text.perLayerConfig ?? [:]
        var overridesByLayer: [Int: LayerOverride] = [:]
        for (key, value) in overrides {
            guard let index = Int(key) else {
                throw DecodingError.dataCorruptedError(
                    forKey: RootKeys.textConfig, in: root,
                    debugDescription: "Per-layer keys must be layer indices.")
            }
            overridesByLayer[index] = value
        }
        let hasDefaultRopes = text.ropeParameters.values.allSatisfy {
            ($0.ropeType ?? "default") == "default"
        }
        guard layerTypes.count == hiddenLayers,
            hiddenLayers > 0, slidingWindow > 0, hasDefaultRopes
        else {
            throw DecodingError.dataCorruptedError(
                forKey: RootKeys.textConfig, in: root,
                debugDescription: "Unsupported EmbeddingGemma 2 layer layout.")
        }
        var attention: [Attention] = []
        attention.reserveCapacity(hiddenLayers)
        for (index, type) in layerTypes.enumerated() {
            guard let rope = text.ropeParameters[type] else {
                throw DecodingError.dataCorruptedError(
                    forKey: RootKeys.textConfig, in: root,
                    debugDescription: "Missing RoPE parameters for \(type).")
            }
            let override = overridesByLayer[index]
            attention.append(
                Attention(
                    headDim: override?.headDim ?? headDim,
                    kvHeads: override?.keyValueHeads ?? kvHeads,
                    ropeBase: rope.ropeTheta,
                    isGlobal: type == "full_attention"))
        }
        self.attention = attention

        // The library covers the encoder layout EmbeddingGemma 2 ships: no activation
        // clipping and no standardization.
        if let encoder = try root.decodeIfPresent(
            Gemma4VisionConfiguration.self, forKey: .visionConfig),
            !encoder.useClippedLinears, !encoder.standardize,
            let image = try root.decodeIfPresent(Int.self, forKey: .imageTokenId),
            let begin = try root.decodeIfPresent(Int.self, forKey: .boiTokenId),
            let end = try root.decodeIfPresent(Int.self, forKey: .eoiTokenId)
        {
            vision = Vision(
                encoder: encoder, imageTokenID: image,
                videoTokenID: try root.decodeIfPresent(Int.self, forKey: .videoTokenId),
                beginImageTokenID: begin, endImageTokenID: end)
        }
        // `embedding_gemma2` checkpoints name the end marker `eoa_token_index`.
        if let encoder = try root.decodeIfPresent(
            Gemma4AudioConfiguration.self, forKey: .audioConfig),
            let token = try root.decodeIfPresent(Int.self, forKey: .audioTokenId),
            let begin = try root.decodeIfPresent(Int.self, forKey: .boaTokenId),
            let end = try root.decodeIfPresent(Int.self, forKey: .eoaTokenId)
                ?? root.decodeIfPresent(Int.self, forKey: .eoaTokenIndex)
        {
            audio = Audio(
                encoder: encoder, audioTokenID: token, beginAudioTokenID: begin,
                endAudioTokenID: end)
        }
    }

    public func encode(to encoder: any Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(hiddenSize, forKey: .hiddenSize)
        try container.encode(hiddenLayers, forKey: .hiddenLayers)
        try container.encode(attentionHeads, forKey: .attentionHeads)
        try container.encode(kvHeads, forKey: .kvHeads)
        try container.encode(headDim, forKey: .headDim)
        try container.encode(intermediateSize, forKey: .intermediateSize)
        try container.encode(perLayerInputSize, forKey: .perLayerInputSize)
        try container.encode(embeddingDim, forKey: .embeddingDim)
        try container.encode(vocabularySize, forKey: .vocabularySize)
        try container.encode(rmsNormEps, forKey: .rmsNormEps)
        try container.encode(slidingWindow, forKey: .slidingWindow)
        try container.encode(layerTypes, forKey: .layerTypes)
    }

    private enum CodingKeys: String, CodingKey {
        case hiddenSize = "hidden_size"
        case hiddenLayers = "num_hidden_layers"
        case attentionHeads = "num_attention_heads"
        case kvHeads = "num_key_value_heads"
        case headDim = "head_dim"
        case intermediateSize = "intermediate_size"
        case perLayerInputSize = "hidden_size_per_layer_input"
        case embeddingDim = "embedding_dim"
        case vocabularySize = "vocab_size"
        case rmsNormEps = "rms_norm_eps"
        case slidingWindow = "sliding_window"
        case layerTypes = "layer_types"
    }
}

// MARK: - Model

/// EmbeddingGemma 2: a bidirectional text encoder with per-layer embeddings that maps
/// text and the soft tokens of images, video frames and audio into one normalized
/// ``EmbeddingGemma2Configuration/embeddingDim`` space. Load checkpoints with
/// ``loadWeights(modelDirectory:model:quantization:perLayerQuantization:)``.
public final class EmbeddingGemma2: Module, BaseLanguageModel {

    public let config: EmbeddingGemma2Configuration

    @ModuleInfo(key: "language_model") private var languageModel: EmbeddingGemma2TextModel
    @ModuleInfo(key: "vision_tower") private var visionTower: Gemma4VisionModel?
    @ModuleInfo(key: "embed_vision") private var embedVision: Gemma4MultimodalEmbedder?
    @ModuleInfo(key: "audio_tower") private var audioTower: Gemma4AudioModel?
    @ModuleInfo(key: "embed_audio") private var embedAudio: Gemma4MultimodalEmbedder?

    private let softTokenIDs: [Int32]

    public init(_ config: EmbeddingGemma2Configuration) {
        self.config = config
        self._languageModel.wrappedValue = EmbeddingGemma2TextModel(config)
        if let vision = config.vision {
            self._visionTower.wrappedValue = Gemma4VisionModel(config: vision.encoder)
            self._embedVision.wrappedValue = Gemma4MultimodalEmbedder(
                embeddingDim: vision.encoder.hiddenSize, textHiddenSize: config.hiddenSize,
                eps: vision.encoder.rmsNormEps)
        }
        if let audio = config.audio {
            self._audioTower.wrappedValue = Gemma4AudioModel(config: audio.encoder)
            self._embedAudio.wrappedValue = Gemma4MultimodalEmbedder(
                embeddingDim: audio.encoder.outputProjectionDimensions,
                textHiddenSize: config.hiddenSize, eps: audio.encoder.rmsNormEps)
        }
        self.softTokenIDs = config.softTokenIDs.map(Int32.init)
        super.init()
    }

    /// - Parameter pixels: `[B, 3, H, W]` images or video frames of one size.
    /// - Returns: `[B, soft tokens, hidden]` features that fill their placeholder tokens,
    ///   or `nil` when this checkpoint has no vision encoder.
    public func imageFeatures(_ pixels: MLXArray) -> MLXArray? {
        guard let visionTower, let embedVision else { return nil }
        return embedVision(visionTower(pixels))
    }

    /// - Parameter features: `[frames, featureSize]` log-mel features of one audio.
    /// - Returns: `[1, soft tokens, hidden]` features that fill its placeholder tokens, or
    ///   `nil` when this checkpoint has no audio encoder.
    public func audioFeatures(_ features: MLXArray) -> MLXArray? {
        guard let audioTower, let embedAudio else { return nil }
        return embedAudio(audioTower(features))
    }

    /// - Parameters:
    ///   - inputIds: `[B, L]` tokens.
    ///   - attentionMask: `[B, L]`, `1` for real tokens. Attention and pooling ignore padding.
    ///   - softTokens: `[soft tokens, hidden]` features of every image, video and audio
    ///     placeholder in `inputIds`, in sequence order; single row only.
    /// - Returns: `[B, embeddingDim]` unit-length float32 embeddings.
    public func embed(
        inputIds: MLXArray, attentionMask: MLXArray?, softTokens: MLXArray? = nil
    ) -> MLXArray {
        precondition(inputIds.ndim == 2 && inputIds.dim(1) > 0)
        if let attentionMask {
            precondition(attentionMask.shape == inputIds.shape)
        }
        var hidden = languageModel.embedTokens(inputIds)
        // The model card forbids float16: activations exceed its range.
        if hidden.dtype == .float16 { hidden = hidden.asType(.float32) }
        // Matches the reference: sqrt(hidden) is rounded to the weight dtype (22.625 in bf16).
        hidden = hidden * MLXArray(Float(config.hiddenSize).squareRoot()).asType(hidden.dtype)
        if let softTokens, let first = softTokenIDs.first {
            precondition(inputIds.dim(0) == 1)
            let isSoftToken = softTokenIDs.dropFirst().reduce(inputIds .== first) {
                $0 .|| (inputIds .== $1)
            }
            let index = maximum(cumsum(isSoftToken.asType(.int32), axis: 1) - 1, 0)
            let rows = softTokens.asType(hidden.dtype).reshaped(-1, hidden.dim(-1))
                .take(index.squeezed(axis: 0), axis: 0).expandedDimensions(axis: 0)
            hidden = MLX.where(isSoftToken.expandedDimensions(axis: -1), rows, hidden)
        }
        hidden = languageModel(hidden, attentionMask: attentionMask)

        // The projection is linear and bias-free, so pooling first is exact and cheaper.
        let pooled = meanPooling(hiddenStates: hidden, attentionMask: attentionMask)
        let projected = languageModel.embeddingProjection(pooled.asType(hidden.dtype))
        let vector = projected.asType(.float32)
        let norm = maximum(
            MLX.sqrt((vector * vector).sum(axis: -1, keepDims: true)), MLXArray(1e-9))
        return vector / norm
    }

    /// Keeps the text model and the encoders this configuration loads. PyTorch checkpoints
    /// store convolution kernels channels-first; they move to the channels-last layout
    /// of MLX.
    public func sanitize(weights: [String: MLXArray]) throws -> [String: MLXArray] {
        var prefixes = ["language_model."]
        if visionTower != nil { prefixes += ["vision_tower.", "embed_vision."] }
        if audioTower != nil { prefixes += ["audio_tower.", "embed_audio."] }
        let shapes = Dictionary(
            uniqueKeysWithValues: parameters().flattened().map { ($0.0, $0.1.shape) })
        var clean: [String: MLXArray] = [:]
        for (key, value) in weights where prefixes.contains(where: key.hasPrefix) {
            if let shape = shapes[key], value.ndim > 2, value.shape != shape {
                let channelsLast = value.movedAxis(source: 1, destination: -1)
                clean[key] = channelsLast.shape == shape ? channelsLast : value
            } else {
                clean[key] = value
            }
        }
        return clean
    }
}

/// Mean pooling over the attention mask; every token counts when no mask is given.
private func meanPooling(hiddenStates: MLXArray, attentionMask: MLXArray?) -> MLXArray {
    guard let mask = attentionMask else { return hiddenStates.mean(axis: 1) }
    let expanded = mask.expandedDimensions(axes: [2]).asType(.float32)
    let sum = (hiddenStates.asType(.float32) * expanded).sum(axis: 1)
    return sum / maximum(expanded.sum(axis: 1), MLXArray(1e-9))
}

// MARK: - Text Encoder

/// The bidirectional text model with projection-only per-layer embeddings (PLE).
private final class EmbeddingGemma2TextModel: Module {

    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    @ModuleInfo(key: "ple") var ple: EmbeddingGemma2PLE
    @ModuleInfo(key: "layers") var layers: [EmbeddingGemma2Layer]
    @ModuleInfo(key: "norm") var norm: EmbeddingGemma2RMSNorm
    @ModuleInfo(key: "embedding_projection") var embeddingProjection: Linear

    private let slidingWindow: Int

    init(_ config: EmbeddingGemma2Configuration) {
        self.slidingWindow = config.slidingWindow
        self._embedTokens.wrappedValue = Embedding(
            embeddingCount: config.vocabularySize, dimensions: config.hiddenSize)
        self._ple.wrappedValue = EmbeddingGemma2PLE(config)
        self._layers.wrappedValue = config.attention.map {
            EmbeddingGemma2Layer(config, attention: $0)
        }
        self._norm.wrappedValue = EmbeddingGemma2RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
        self._embeddingProjection.wrappedValue = Linear(
            config.hiddenSize, config.embeddingDim, bias: false)
    }

    func callAsFunction(_ x: MLXArray, attentionMask: MLXArray?) -> MLXArray {
        let perLayerInputs = ple(x)

        let (batch, length) = (x.dim(0), x.dim(1))
        let globalMask = EmbeddingGemma2Masks.bidirectional(
            batch: batch, seqLen: length, paddingMask: attentionMask)
        // The window pattern equals the full mask while the sequence fits inside it.
        let localMask =
            length - 1 > slidingWindow
            ? EmbeddingGemma2Masks.combine(
                pattern: EmbeddingGemma2Masks.slidingWindowPattern(
                    seqLen: length, radius: slidingWindow),
                batch: batch, seqLen: length, paddingMask: attentionMask)
            : globalMask

        var hidden = x
        for (index, layer) in layers.enumerated() {
            hidden = layer(
                hidden,
                perLayerInput: perLayerInputs[0..., 0..., index, 0...],
                mask: layer.isGlobal ? globalMask : localMask)
        }
        return norm(hidden)
    }
}

/// Derives every layer's gating signal from the scaled token embeddings alone.
private final class EmbeddingGemma2PLE: Module {
    let layerCount: Int
    let inputSize: Int
    let scale: Float

    @ModuleInfo(key: "per_layer_model_projection") var perLayerModelProjection: Linear
    @ModuleInfo(key: "per_layer_projection_norm") var perLayerProjectionNorm: EmbeddingGemma2RMSNorm

    init(_ config: EmbeddingGemma2Configuration) {
        self.layerCount = config.hiddenLayers
        self.inputSize = config.perLayerInputSize
        self.scale = 1 / Float(config.hiddenSize).squareRoot()
        self._perLayerModelProjection.wrappedValue = Linear(
            config.hiddenSize, layerCount * inputSize, bias: false)
        self._perLayerProjectionNorm.wrappedValue = EmbeddingGemma2RMSNorm(
            dimensions: inputSize, eps: config.rmsNormEps)
    }

    /// - Returns: `[B, L, layers, inputSize]`.
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let projected = perLayerModelProjection(x) * scale
        return perLayerProjectionNorm(
            projected.reshaped(x.dim(0), x.dim(1), layerCount, inputSize))
    }
}

private final class EmbeddingGemma2Layer: Module {
    @ModuleInfo(key: "self_attn") var selfAttention: EmbeddingGemma2Attention
    @ModuleInfo(key: "mlp") var mlp: EmbeddingGemma2MLP
    @ModuleInfo(key: "ple_block") var pleBlock: EmbeddingGemma2PLEBlock
    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: EmbeddingGemma2RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: EmbeddingGemma2RMSNorm
    @ModuleInfo(key: "pre_feedforward_layernorm") var preFeedforwardLayerNorm:
        EmbeddingGemma2RMSNorm
    @ModuleInfo(key: "post_feedforward_layernorm") var postFeedforwardLayerNorm:
        EmbeddingGemma2RMSNorm
    @ModuleInfo(key: "layer_scalar") var layerScalar: MLXArray

    let isGlobal: Bool

    init(_ config: EmbeddingGemma2Configuration, attention: EmbeddingGemma2Configuration.Attention)
    {
        let (size, eps) = (config.hiddenSize, config.rmsNormEps)
        self.isGlobal = attention.isGlobal
        self._selfAttention.wrappedValue = EmbeddingGemma2Attention(config, attention: attention)
        self._mlp.wrappedValue = EmbeddingGemma2MLP(config)
        self._pleBlock.wrappedValue = EmbeddingGemma2PLEBlock(config)
        self._inputLayerNorm.wrappedValue = EmbeddingGemma2RMSNorm(dimensions: size, eps: eps)
        self._postAttentionLayerNorm.wrappedValue = EmbeddingGemma2RMSNorm(
            dimensions: size, eps: eps)
        self._preFeedforwardLayerNorm.wrappedValue = EmbeddingGemma2RMSNorm(
            dimensions: size, eps: eps)
        self._postFeedforwardLayerNorm.wrappedValue = EmbeddingGemma2RMSNorm(
            dimensions: size, eps: eps)
        self._layerScalar.wrappedValue = MLXArray.ones([1])
    }

    func callAsFunction(_ x: MLXArray, perLayerInput: MLXArray, mask: MLXArray?) -> MLXArray {
        var hidden = x + postAttentionLayerNorm(selfAttention(inputLayerNorm(x), mask: mask))
        hidden = hidden + postFeedforwardLayerNorm(mlp(preFeedforwardLayerNorm(hidden)))
        return pleBlock(hidden, perLayerInput: perLayerInput) * layerScalar
    }
}

private final class EmbeddingGemma2Attention: Module {
    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "o_proj") var oProj: Linear
    @ModuleInfo(key: "q_norm") var qNorm: EmbeddingGemma2RMSNorm
    @ModuleInfo(key: "k_norm") var kNorm: EmbeddingGemma2RMSNorm

    let rope: MLXNN.RoPE
    let heads: Int
    let kvHeads: Int
    let headDim: Int
    private let eps: Float

    init(_ config: EmbeddingGemma2Configuration, attention: EmbeddingGemma2Configuration.Attention)
    {
        self.heads = config.attentionHeads
        self.kvHeads = attention.kvHeads
        self.headDim = attention.headDim
        self.eps = config.rmsNormEps
        self.rope = MLXNN.RoPE(dimensions: headDim, traditional: false, base: attention.ropeBase)
        self._qProj.wrappedValue = Linear(config.hiddenSize, heads * headDim, bias: false)
        self._kProj.wrappedValue = Linear(config.hiddenSize, kvHeads * headDim, bias: false)
        self._vProj.wrappedValue = Linear(config.hiddenSize, kvHeads * headDim, bias: false)
        self._oProj.wrappedValue = Linear(heads * headDim, config.hiddenSize, bias: false)
        self._qNorm.wrappedValue = EmbeddingGemma2RMSNorm(dimensions: headDim, eps: eps)
        self._kNorm.wrappedValue = EmbeddingGemma2RMSNorm(dimensions: headDim, eps: eps)
    }

    func callAsFunction(_ x: MLXArray, mask: MLXArray?) -> MLXArray {
        let (batch, length) = (x.dim(0), x.dim(1))
        let q = rope(qNorm(qProj(x).reshaped(batch, length, heads, headDim)).transposed(0, 2, 1, 3))
        let k = rope(
            kNorm(kProj(x).reshaped(batch, length, kvHeads, headDim)).transposed(0, 2, 1, 3))
        let rawValues = vProj(x).reshaped(batch, length, kvHeads, headDim)
        // Values are normalized without a learned scale; the query and key norms replace
        // the 1/sqrt(d) scale.
        let v = MLXFast.rmsNorm(
            rawValues, weight: MLXArray.ones([headDim], dtype: .float32), eps: eps
        )
        .asType(rawValues.dtype)
        .transposed(0, 2, 1, 3)
        let output = MLXFast.scaledDotProductAttention(
            queries: q, keys: k, values: v, scale: 1, mask: mask.map { .array($0) } ?? .none)
        return oProj(output.transposed(0, 2, 1, 3).reshaped(batch, length, -1))
    }
}

private final class EmbeddingGemma2MLP: Module {
    @ModuleInfo(key: "gate_proj") var gateProj: Linear
    @ModuleInfo(key: "up_proj") var upProj: Linear
    @ModuleInfo(key: "down_proj") var downProj: Linear

    init(_ config: EmbeddingGemma2Configuration) {
        self._gateProj.wrappedValue = Linear(
            config.hiddenSize, config.intermediateSize, bias: false)
        self._upProj.wrappedValue = Linear(config.hiddenSize, config.intermediateSize, bias: false)
        self._downProj.wrappedValue = Linear(
            config.intermediateSize, config.hiddenSize, bias: false)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        downProj(geluApproximate(gateProj(x)) * upProj(x))
    }
}

/// Gates the residual stream with this layer's slice of the per-layer embeddings.
private final class EmbeddingGemma2PLEBlock: Module {
    @ModuleInfo(key: "per_layer_input_gate") var perLayerInputGate: Linear
    @ModuleInfo(key: "per_layer_projection") var perLayerProjection: Linear
    @ModuleInfo(key: "post_per_layer_input_norm") var postPerLayerInputNorm: EmbeddingGemma2RMSNorm

    init(_ config: EmbeddingGemma2Configuration) {
        self._perLayerInputGate.wrappedValue = Linear(
            config.hiddenSize, config.perLayerInputSize, bias: false)
        self._perLayerProjection.wrappedValue = Linear(
            config.perLayerInputSize, config.hiddenSize, bias: false)
        self._postPerLayerInputNorm.wrappedValue = EmbeddingGemma2RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
    }

    func callAsFunction(_ x: MLXArray, perLayerInput: MLXArray) -> MLXArray {
        let gated = geluApproximate(perLayerInputGate(x)) * perLayerInput
        return x + postPerLayerInputNorm(perLayerProjection(gated))
    }
}

/// Gemma 4 RMSNorm: the weight scales directly (no `1 + weight` offset).
private final class EmbeddingGemma2RMSNorm: Module {
    let weight: MLXArray
    let eps: Float

    init(dimensions: Int, eps: Float) {
        self.weight = MLXArray.ones([dimensions])
        self.eps = eps
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        MLXFast.rmsNorm(x, weight: weight.asType(.float32), eps: eps).asType(x.dtype)
    }
}

// MARK: - Masks

enum EmbeddingGemma2Masks {

    /// Full bidirectional mask over real tokens; `nil` when nothing is padded. Entirely
    /// padded rows keep attention defined; pooling discards their states.
    static func bidirectional(batch: Int, seqLen: Int, paddingMask: MLXArray?) -> MLXArray? {
        guard let paddingMask else { return nil }
        let validKeys = paddingMask.asType(.bool).reshaped(batch, 1, 1, seqLen)
        return validKeys .|| logicalNot(validKeys.any(axis: -1, keepDims: true))
    }

    /// Keys within `radius` of the query, both endpoints included.
    static func slidingWindowPattern(seqLen: Int, radius: Int) -> MLXArray {
        let rows = MLXArray(0 ..< seqLen).reshaped(seqLen, 1)
        let columns = MLXArray(0 ..< seqLen).reshaped(1, seqLen)
        return (abs(rows - columns) .<= MLXArray(Int32(radius))).reshaped(1, 1, seqLen, seqLen)
    }

    /// Combines the window pattern with a padding mask, keeping the diagonal so a padded
    /// query never has a fully masked attention row.
    static func combine(pattern: MLXArray, batch: Int, seqLen: Int, paddingMask: MLXArray?)
        -> MLXArray
    {
        guard let paddingMask else { return pattern }
        let validKeys = paddingMask.asType(.bool).reshaped(batch, 1, 1, seqLen)
        let diagonal =
            (MLXArray(0 ..< seqLen).reshaped(seqLen, 1)
            .== MLXArray(0 ..< seqLen).reshaped(1, seqLen))
            .reshaped(1, 1, seqLen, seqLen)
        return (pattern .&& validKeys) .|| diagonal
    }
}

// MARK: - Image Preparation

/// Prepares images as the reference `Gemma4ImageProcessor` does: the target size and
/// soft-token budget of ``Gemma4ProcessorConfiguration``, with an antialiased bicubic
/// resize. The Core Image resample moves a resized image's embedding to cosine 0.995
/// from the reference; this one keeps it above 0.9999.
public struct EmbeddingGemma2ImageProcessor: Sendable {

    public let configuration: Gemma4ProcessorConfiguration

    public init(directory: URL) throws {
        self.configuration = try JSONDecoder().decode(
            Gemma4ProcessorConfiguration.self,
            from: try Data(contentsOf: directory.appendingPathComponent("processor_config.json")))
    }

    /// - Returns: `[1, 3, H, W]` pixels in `0...1` resized to the soft-token budget,
    ///   and their soft-token count.
    public func pixels(for image: UserInput.Image) throws -> (pixels: MLXArray, tokens: Int) {
        let image = try Self.prepared(image.asCIImage())
        let target = configuration.aspectPreservingTargetSize(for: image.extent.size)
        let (height, width) = (Int(target.height), Int(target.width))
        return (
            Self.pixels(image, height: height, width: width),
            configuration.softTokenCount(height: height, width: width)
        )
    }

    /// The oriented image on the sRGB tone curve, as the reference reads it.
    static func prepared(_ image: CIImage) throws -> CIImage {
        let oriented = image.settingProperties([CIImageOption.applyOrientationProperty: true])
        guard !oriented.extent.isEmpty, !oriented.extent.isInfinite else {
            throw EmbeddingGemma2Embedding.Error.invalidMedia
        }
        return MediaProcessing.inSRGBToneCurveSpace(oriented)
    }

    /// `[1, 3, height, width]` pixels in `0...1`.
    static func pixels(_ image: CIImage, height: Int, width: Int) -> MLXArray {
        let pixels = MediaProcessing.asMLXArray(image)
        guard pixels.dim(2) != height || pixels.dim(3) != width else { return pixels }
        return resized(pixels, height: height, width: width)
    }

    /// Separable resize as two matrix products, rounded to 8 bits like the reference's
    /// `uint8` output.
    static func resized(_ pixels: MLXArray, height: Int, width: Int) -> MLXArray {
        let rows = bicubicWeights(input: pixels.dim(2), output: height)
        let columns = bicubicWeights(input: pixels.dim(3), output: width)
        let resized = matmul(matmul(rows, pixels), columns.transposed())
        return round(clip(resized * 255, min: 0, max: 255)) / 255
    }

    /// `[output, input]` interpolation weights of the antialiased bicubic filter
    /// (a = -0.5) of Pillow and torchvision.
    static func bicubicWeights(input: Int, output: Int) -> MLXArray {
        let scale = Double(input) / Double(output)
        let filterScale = max(scale, 1)
        let support = 2 * filterScale
        var weights = [Float](repeating: 0, count: output * input)
        for row in 0 ..< output {
            let center = (Double(row) + 0.5) * scale
            let first = max(Int(center - support + 0.5), 0)
            let last = min(Int(center + support + 0.5), input)
            let taps = (first ..< last).map { cubic((Double($0) - center + 0.5) / filterScale) }
            let total = taps.reduce(0, +)
            for (offset, tap) in taps.enumerated() where total != 0 {
                weights[row * input + first + offset] = Float(tap / total)
            }
        }
        return MLXArray(weights, [output, input])
    }

    private static func cubic(_ x: Double) -> Double {
        let a = -0.5
        let x = abs(x)
        if x < 1 { return ((a + 2) * x - (a + 3)) * x * x + 1 }
        if x < 2 { return (((x - 5) * x + 8) * x - 4) * a }
        return 0
    }
}

// MARK: - Video Preparation

/// Prepares video as the reference `EmbeddingGemma2VideoProcessor` does: one frame per
/// second, spread evenly over at most ``maximumFrames``, each resized like an image to the
/// smaller frame budget. The audio track is not read.
public struct EmbeddingGemma2VideoProcessor: Sendable {

    /// Frames sampled per second of video.
    public let framesPerSecond: Double
    public let maximumFrames: Int
    /// Soft tokens per frame.
    public let budget: Int

    private let images: EmbeddingGemma2ImageProcessor

    private struct Configuration: Decodable {
        struct VideoProcessor: Decodable {
            let fps: Double
            let maxFrames: Int
            let maxSoftTokens: Int
            let overflowStrategy: String
            let addTimestamps: Bool
        }
        let videoProcessor: VideoProcessor
    }

    /// Reads `video_processor` from `processor_config.json`. Throws unless frames spread
    /// evenly and carry no timestamps, the only layout the library builds.
    public init(directory: URL) throws {
        let decoder = JSONDecoder()
        decoder.keyDecodingStrategy = .convertFromSnakeCase
        let video = try decoder.decode(
            Configuration.self,
            from: try Data(contentsOf: directory.appendingPathComponent("processor_config.json"))
        ).videoProcessor
        guard video.overflowStrategy == "uniform", !video.addTimestamps, video.fps > 0,
            video.maxFrames > 0
        else {
            throw DecodingError.dataCorrupted(
                .init(codingPath: [], debugDescription: "Unsupported video sampling."))
        }
        self.framesPerSecond = video.fps
        self.maximumFrames = video.maxFrames
        self.budget = video.maxSoftTokens
        self.images = try EmbeddingGemma2ImageProcessor(directory: directory)
    }

    /// - Returns: `[frames, 3, H, W]` pixels in `0...1`, every frame at the size of the first.
    public nonisolated(nonsending) func frames(for video: UserInput.Video) async throws
        -> MLXArray
    {
        let frames = try await sampledFrames(video).map(EmbeddingGemma2ImageProcessor.prepared)
        guard let first = frames.first else { throw EmbeddingGemma2Embedding.Error.invalidMedia }
        let target = images.configuration.aspectPreservingTargetSize(
            for: first.extent.size, budget: budget)
        let (height, width) = (Int(target.height), Int(target.width))
        return concatenated(
            frames.map { EmbeddingGemma2ImageProcessor.pixels($0, height: height, width: width) },
            axis: 0)
    }

    private nonisolated(nonsending) func sampledFrames(_ video: UserInput.Video) async throws
        -> [CIImage]
    {
        switch video.source {
        case .frames(let frames):
            // Decoded frames carry no frame rate, so all of them count before the cap.
            return try Self.sampledIndices(
                frameCount: frames.count, frameRate: nil, framesPerSecond: framesPerSecond,
                maximumFrames: maximumFrames
            ).map { try frames[$0].image.asCIImage() }
        case .url(let url):
            return try await sampledFrames(AVURLAsset(url: url))
        case .avAsset(let asset):
            return try await sampledFrames(asset)
        }
    }

    private nonisolated(nonsending) func sampledFrames(_ asset: AVAsset) async throws -> [CIImage] {
        guard let track = try await asset.loadTracks(withMediaType: .video).first else {
            throw EmbeddingGemma2Embedding.Error.invalidMedia
        }
        let (range, nominalRate) = try await track.load(.timeRange, .nominalFrameRate)
        let frameRate = Double(nominalRate)
        guard frameRate > 0 else { throw EmbeddingGemma2Embedding.Error.invalidMedia }
        let indices = Self.sampledIndices(
            frameCount: Int((range.duration.seconds * frameRate).rounded()),
            frameRate: frameRate, framesPerSecond: framesPerSecond, maximumFrames: maximumFrames)

        let generator = AVAssetImageGenerator(asset: asset)
        generator.appliesPreferredTrackTransform = true
        generator.requestedTimeToleranceBefore = .zero
        generator.requestedTimeToleranceAfter = .zero
        // The middle of each frame's interval selects it without rounding to a neighbor.
        let times = indices.map {
            CMTime(
                seconds: range.start.seconds + (Double($0) + 0.5) / frameRate,
                preferredTimescale: 90_000)
        }
        var frames: [CIImage] = []
        frames.reserveCapacity(times.count)
        // Like the decoders of the reference, read the decoded RGB values as sRGB.
        let options: [CIImageOption: Any] =
            CGColorSpace(name: CGColorSpace.sRGB).map { [.colorSpace: $0] } ?? [:]
        for await result in generator.images(for: times) {
            if case .success(_, let image, _) = result {
                frames.append(CIImage(cgImage: image, options: options))
            }
        }
        return frames
    }

    /// Indices of the reference `sample_frames`: one frame every `frameRate /
    /// framesPerSecond` frames, then `maximumFrames` spread evenly over them, as
    /// `np.linspace` truncates. Without a frame rate every frame is a candidate.
    static func sampledIndices(
        frameCount: Int, frameRate: Double?, framesPerSecond: Double, maximumFrames: Int
    ) -> [Int] {
        var indices = Array(0 ..< frameCount)
        if let frameRate, frameCount > 0 {
            let step = frameRate / framesPerSecond
            let count = max(1, Int(Double(frameCount) / frameRate * framesPerSecond))
            indices = (0 ..< count).map { min(frameCount - 1, Int(Double($0) * step)) }
        }
        guard indices.count > maximumFrames else { return indices }
        let step = Double(indices.count - 1) / Double(max(maximumFrames - 1, 1))
        return (0 ..< maximumFrames).map { position in
            position > 0 && position == maximumFrames - 1
                ? indices[indices.count - 1] : indices[Int(Double(position) * step)]
        }
    }
}

// MARK: - Audio Preparation

/// Log-mel features as the reference `Gemma4AudioFeatureExtractor` computes them: a
/// semicausal short-time Fourier transform under a periodic Hann window, an HTK mel
/// filter bank, and a log floor. Only frames of real samples are kept, so the audio tower
/// runs without padding, and audio past ``maximumSampleCount`` is dropped.
public struct EmbeddingGemma2AudioProcessor: Sendable {

    /// The reference keeps the first 480,000 samples: 30 seconds at 16 kHz.
    public static let maximumSampleCount = 480_000

    /// Samples per second of the mono audio the features read.
    public let sampleRate: Int
    private let frameLength: Int
    private let hopLength: Int
    private let fftLength: Int
    private let featureSize: Int
    private let melFloor: Float
    /// Periodic Hann window over one frame.
    private let window: [Float]
    /// `[fftLength / 2 + 1, featureSize]` triangular filters, row-major.
    private let melFilters: [Float]

    private struct Configuration: Decodable {
        struct FeatureExtractor: Decodable {
            let featureSize: Int
            let samplingRate: Int
            let frameLength: Int
            let hopLength: Int
            let fftLength: Int
            let minFrequency: Double
            let maxFrequency: Double
            let melFloor: Float
            let preemphasis: Double?
            let inputScaleFactor: Double?
            let perBinMean: [Double]?
            let perBinStddev: [Double]?
        }
        let featureExtractor: FeatureExtractor
    }

    /// Reads `feature_extractor` from `processor_config.json`. Throws when it asks for
    /// pre-emphasis, input scaling or per-bin normalization, which the checkpoint does not use.
    public init(directory: URL) throws {
        let decoder = JSONDecoder()
        decoder.keyDecodingStrategy = .convertFromSnakeCase
        let extractor = try decoder.decode(
            Configuration.self,
            from: try Data(contentsOf: directory.appendingPathComponent("processor_config.json"))
        ).featureExtractor
        guard (extractor.preemphasis ?? 0) == 0, (extractor.inputScaleFactor ?? 1) == 1,
            extractor.perBinMean == nil, extractor.perBinStddev == nil,
            extractor.frameLength <= extractor.fftLength, extractor.hopLength > 0
        else {
            throw DecodingError.dataCorrupted(
                .init(codingPath: [], debugDescription: "Unsupported audio feature extractor."))
        }
        self.sampleRate = extractor.samplingRate
        self.frameLength = extractor.frameLength
        self.hopLength = extractor.hopLength
        self.fftLength = extractor.fftLength
        self.featureSize = extractor.featureSize
        self.melFloor = extractor.melFloor
        self.window = (0 ..< extractor.frameLength).map {
            Float(0.5 - 0.5 * cos(2 * Double.pi * Double($0) / Double(extractor.frameLength)))
        }
        self.melFilters = Self.melFilters(
            bins: extractor.fftLength / 2 + 1, mels: extractor.featureSize,
            minFrequency: extractor.minFrequency, maxFrequency: extractor.maxFrequency,
            sampleRate: extractor.samplingRate)
    }

    /// Decodes mono audio at ``sampleRate``; `.array` sources must already be.
    ///
    /// - Returns: `[frames, featureSize]` log-mel features. Audio shorter than one frame
    ///   has none.
    public nonisolated(nonsending) func features(for audio: UserInput.Audio) async throws
        -> MLXArray
    {
        let samples: MLXArray
        switch audio.source {
        case .array(let array):
            samples = array
        case .url(let url):
            var processing = UserInput.AudioProcessing()
            processing.sampleRate = Double(sampleRate)
            // A new value, not the caller's, crosses into the concurrent decoder.
            samples = try await UserInput.Audio.url(url).asMLXArray(processing: processing)
        }
        guard samples.ndim == 1 else { throw EmbeddingGemma2Embedding.Error.invalidMedia }
        return features(samples: samples)
    }

    /// `[frames, featureSize]` features of the frames that hold only real samples.
    func features(samples: MLXArray) -> MLXArray {
        let samples = samples[..<min(samples.dim(0), Self.maximumSampleCount)].asType(.float32)
        // Semicausal padding centres the first frame on the first sample.
        let padded = concatenated([MLXArray.zeros([frameLength / 2]), samples])
        // Each reference frame spans one sample more than it transforms.
        let span = frameLength + 1
        guard padded.dim(0) >= span else { return MLXArray.zeros([0, featureSize]) }
        let frames = asStrided(
            padded, [(padded.dim(0) - span) / hopLength + 1, frameLength],
            strides: [hopLength, 1])
        let magnitudes = abs(MLXFFT.rfft(frames * MLXArray(window), n: fftLength, axis: -1))
        let mel = matmul(magnitudes, MLXArray(melFilters, [fftLength / 2 + 1, featureSize]))
        return log(mel + melFloor)
    }

    /// The reference `mel_filter_bank` with HTK mels and no normalization.
    static func melFilters(
        bins: Int, mels: Int, minFrequency: Double, maxFrequency: Double, sampleRate: Int
    ) -> [Float] {
        func mel(_ hertz: Double) -> Double { 2595 * log10(1 + hertz / 700) }
        func hertz(_ mel: Double) -> Double { 700 * (pow(10, mel / 2595) - 1) }
        let (low, high) = (mel(minFrequency), mel(maxFrequency))
        let edges = (0 ... mels + 1).map {
            hertz(low + Double($0) * (high - low) / Double(mels + 1))
        }
        let nyquist = Double(sampleRate / 2)
        var filters = [Float](repeating: 0, count: bins * mels)
        for bin in 0 ..< bins {
            let frequency = nyquist * Double(bin) / Double(bins - 1)
            for filter in 0 ..< mels {
                let rising = (frequency - edges[filter]) / (edges[filter + 1] - edges[filter])
                let falling =
                    (edges[filter + 2] - frequency) / (edges[filter + 2] - edges[filter + 1])
                filters[bin * mels + filter] = Float(max(0, min(rising, falling)))
            }
        }
        return filters
    }
}

// MARK: - Sequence Layout

/// Builds the token sequence of one input, as the reference processor and chat template do.
public enum EmbeddingGemma2Sequence {

    /// One run of an input: text tokens or the soft tokens of one media item.
    public enum Segment: Equatable, Sendable {
        case text([Int])
        case image(tokens: Int)
        case video(frames: Int, tokensPerFrame: Int)
        case audio(tokens: Int)
    }

    /// - Returns: `<bos>`, the segments in order, then `<eos>`. An image and each video
    ///   frame become `<boi> <placeholder>×n <eoi>`, an audio `<boa> <audio>×n <eoa>`.
    ///   Text loses its last tokens to fit `limit`; media is never cut.
    public static func tokens(
        _ segments: [Segment], configuration: EmbeddingGemma2Configuration,
        beginOfSequence: Int, endOfSequence: Int, limit: Int
    ) throws -> [Int] {
        let blocks = try segments.map { try block($0, configuration) }
        var textBudget = limit - 2 - blocks.reduce(0) { $0 + ($1?.count ?? 0) }
        guard textBudget >= 0 else { throw EmbeddingGemma2Embedding.Error.contextExceeded }
        let placeholders = Set(configuration.softTokenIDs)
        var tokens = [beginOfSequence]
        for (segment, block) in zip(segments, blocks) {
            if let block {
                tokens += block
            } else if case .text(let text) = segment {
                // A placeholder in text would take the soft tokens of a media item.
                if blocks.contains(where: { $0 != nil }),
                    text.contains(where: placeholders.contains)
                {
                    throw EmbeddingGemma2Embedding.Error.placeholderInText
                }
                tokens += text.prefix(textBudget)
                textBudget -= min(text.count, textBudget)
            }
        }
        tokens.append(endOfSequence)
        return tokens
    }

    /// The placeholder block of a media segment; `nil` for text.
    private static func block(_ segment: Segment, _ configuration: EmbeddingGemma2Configuration)
        throws -> [Int]?
    {
        func marked(_ begin: Int, _ token: Int, _ count: Int, _ end: Int) -> [Int] {
            [begin] + repeatElement(token, count: count) + [end]
        }
        switch segment {
        case .text:
            return nil
        case .image(let count):
            guard let vision = configuration.vision else {
                throw EmbeddingGemma2Embedding.Error.unsupportedMedia
            }
            return marked(
                vision.beginImageTokenID, vision.imageTokenID, count, vision.endImageTokenID)
        case .video(let frames, let count):
            guard let vision = configuration.vision, let token = vision.videoTokenID else {
                throw EmbeddingGemma2Embedding.Error.unsupportedMedia
            }
            let frame = marked(vision.beginImageTokenID, token, count, vision.endImageTokenID)
            return Array(repeatElement(frame, count: frames).joined())
        case .audio(let count):
            guard let audio = configuration.audio else {
                throw EmbeddingGemma2Embedding.Error.unsupportedMedia
            }
            return marked(
                audio.beginAudioTokenID, audio.audioTokenID, count, audio.endAudioTokenID)
        }
    }
}

// MARK: - Embedding

/// Embeddings of text, images, video and audio from an EmbeddingGemma 2 checkpoint, in
/// one shared space. Load the checkpoint once and reuse this actor for every call.
///
/// ```swift
/// let embeddings = try await EmbeddingGemma2Embedding(
///     modelDirectory: directory, tokenizerLoader: loader)
/// let query = try await embeddings.embed(
///     .init(text: "What causes the northern lights?"), task: .searchQuery)
/// let clip = try await embeddings.embed(
///     .init([.text("Aurora over Tromsø: "), .video(.url(movie)), .audio(.url(narration))]),
///     task: .document)
/// ```
public actor EmbeddingGemma2Embedding {

    /// One independently embedded item: text and media in reading order.
    public struct Input {

        /// One piece of an input.
        public enum Part {
            case text(String)
            case image(UserInput.Image)
            /// One frame per second, at most 32, without the audio track.
            case video(UserInput.Video)
            /// Mono audio; the first 30 seconds count.
            case audio(UserInput.Audio)
        }

        public let parts: [Part]
        /// A document title or file name; used by the `.document` task.
        public let title: String?

        public init(_ parts: [Part], title: String? = nil) {
            self.parts = parts
            self.title = title
        }

        /// Text followed by images.
        public init(text: String = "", title: String? = nil, images: [UserInput.Image] = []) {
            self.init([.text(text)] + images.map(Part.image), title: title)
        }
    }

    /// The task an embedding is prepared for. Use `.searchQuery` for queries and
    /// `.document` for corpus items; compared items must share a task's space.
    public enum Task: Sendable {
        case searchQuery
        case document
        case clustering
        case classification
    }

    public enum Error: Swift.Error, LocalizedError, Sendable {
        case emptyInput
        case unsupportedMedia
        case invalidMedia
        case placeholderInText
        case contextExceeded
        case invalidEmbedding

        public var errorDescription: String? {
            switch self {
            case .emptyInput:
                "An embedding input must contain text or media."
            case .unsupportedMedia:
                "This checkpoint cannot embed this kind of media."
            case .invalidMedia:
                "A media item could not be decoded."
            case .placeholderInText:
                "Text next to media must not contain media placeholder tokens."
            case .contextExceeded:
                "The embedding input exceeds \(EmbeddingGemma2Configuration.contextLength) tokens."
            case .invalidEmbedding:
                "The model returned an invalid embedding."
            }
        }
    }

    private let model: EmbeddingGemma2
    private let tokenizer: any Tokenizer
    private let imageProcessor: EmbeddingGemma2ImageProcessor?
    private let videoProcessor: EmbeddingGemma2VideoProcessor?
    private let audioProcessor: EmbeddingGemma2AudioProcessor?

    /// - Parameters:
    ///   - modelDirectory: A local `embedding_gemma2` checkpoint directory.
    ///   - tokenizerLoader: Loads the checkpoint's tokenizer.
    public init(modelDirectory: URL, tokenizerLoader: any TokenizerLoader) async throws {
        let configData = try Data(contentsOf: modelDirectory.appendingPathComponent("config.json"))
        let config = try JSONDecoder().decode(EmbeddingGemma2Configuration.self, from: configData)
        let model = EmbeddingGemma2(config)
        let base = try JSONDecoder().decode(BaseConfiguration.self, from: configData)
        try await loadWeights(
            modelDirectory: modelDirectory, model: model,
            perLayerQuantization: base.perLayerQuantization)
        try Swift.Task.checkCancellation()
        self.model = model
        self.tokenizer = try await tokenizerLoader.load(from: modelDirectory)
        // A missing or unsupported processor configuration disables that media only.
        self.imageProcessor =
            config.vision == nil
            ? nil : try? EmbeddingGemma2ImageProcessor(directory: modelDirectory)
        self.videoProcessor =
            config.vision?.videoTokenID == nil
            ? nil : try? EmbeddingGemma2VideoProcessor(directory: modelDirectory)
        self.audioProcessor =
            config.audio == nil
            ? nil : try? EmbeddingGemma2AudioProcessor(directory: modelDirectory)
    }

    /// Embeds independent items in input order, one at a time to bound working memory.
    /// Use the same task for items that will be compared in the same space.
    public func embed(_ inputs: [Input], task: Task) async throws -> [[Float]] {
        var vectors: [[Float]] = []
        vectors.reserveCapacity(inputs.count)
        for input in inputs {
            try Swift.Task.checkCancellation()
            vectors.append(try await embed(input, task: task))
        }
        return vectors
    }

    /// Embeds one item; returns a unit-length float32 vector.
    ///
    /// Each media item's soft tokens are evaluated before the next one is prepared, so
    /// only one encoder's activations are alive at a time.
    public func embed(_ input: Input, task: Task) async throws -> [Float] {
        let hasText = input.parts.contains { !($0.text ?? "").allSatisfy(\.isWhitespace) }
        guard hasText || input.parts.contains(where: { $0.text == nil }) else {
            throw Error.emptyInput
        }
        guard let begin = tokenizer.bosToken.flatMap({ tokenizer.convertTokenToId($0) }),
            let end = tokenizer.eosTokenId ?? tokenizer.convertTokenToId("<eos>")
        else { throw Error.invalidEmbedding }

        var segments: [EmbeddingGemma2Sequence.Segment] = []
        var softTokens: [MLXArray] = []
        // Adjacent text joins before tokenization, as the chat template renders it.
        var text = ""
        for part in hasText ? Self.prompted(input, task: task) : input.parts {
            if let value = part.text {
                text += value
                continue
            }
            if !text.isEmpty {
                segments.append(.text(tokenizer.encode(text: text, addSpecialTokens: false)))
                text = ""
            }
            let (segment, features) = try await encode(part)
            if let features {
                try MLX.checkedEval(features)
                softTokens.append(features.reshaped(-1, features.dim(-1)))
            }
            segments.append(segment)
            try Swift.Task.checkCancellation()
        }
        if !text.isEmpty {
            segments.append(.text(tokenizer.encode(text: text, addSpecialTokens: false)))
        }

        let tokens = try EmbeddingGemma2Sequence.tokens(
            segments, configuration: model.config, beginOfSequence: begin, endOfSequence: end,
            limit: EmbeddingGemma2Configuration.contextLength)
        let vector = model.embed(
            inputIds: MLXArray(tokens.map(Int32.init), [1, tokens.count]), attentionMask: nil,
            softTokens: softTokens.isEmpty ? nil : concatenated(softTokens, axis: 0))
        try MLX.checkedEval(vector)
        try Swift.Task.checkCancellation()
        let values = vector.asArray(Float.self)
        guard values.allSatisfy(\.isFinite), values.contains(where: { $0 != 0 }) else {
            throw Error.invalidEmbedding
        }
        return values
    }

    /// The segment of one media part and the soft tokens that fill it, `nil` when empty.
    private func encode(_ part: Input.Part) async throws -> (
        EmbeddingGemma2Sequence.Segment, MLXArray?
    ) {
        switch part {
        case .text:
            preconditionFailure("Text parts have no soft tokens.")
        case .image(let image):
            guard let imageProcessor,
                let features = model.imageFeatures(try imageProcessor.pixels(for: image).pixels)
            else { throw Error.unsupportedMedia }
            return (.image(tokens: features.dim(1)), features)
        case .video(let video):
            guard let videoProcessor,
                let features = model.imageFeatures(try await videoProcessor.frames(for: video))
            else { throw Error.unsupportedMedia }
            return (.video(frames: features.dim(0), tokensPerFrame: features.dim(1)), features)
        case .audio(let audio):
            guard let audioProcessor else { throw Error.unsupportedMedia }
            let frames = try await audioProcessor.features(for: audio)
            // Audio shorter than one frame keeps its markers and no soft tokens.
            guard frames.dim(0) > 0 else { return (.audio(tokens: 0), nil) }
            guard let features = model.audioFeatures(frames) else {
                throw Error.unsupportedMedia
            }
            return (.audio(tokens: features.dim(1)), features)
        }
    }

    /// The parts with the task's instruction prefix joined to the leading text.
    private static func prompted(_ input: Input, task: Task) -> [Input.Part] {
        if let first = input.parts.first?.text {
            return [.text(prompt(text: first, title: input.title, task: task))]
                + input.parts.dropFirst()
        }
        return [.text(prompt(text: "", title: input.title, task: task))] + input.parts
    }

    /// The task instruction prefix from the model card. Inputs that already carry one
    /// pass through unchanged.
    public static func prompt(text: String, title: String?, task: Task) -> String {
        if text.hasPrefix("task:") || text.hasPrefix("title:") { return text }
        switch task {
        case .searchQuery:
            return "task: search result | query: \(text)"
        case .clustering:
            return "task: clustering | query: \(text)"
        case .classification:
            return "task: classification | query: \(text)"
        case .document:
            let line = title?
                .split(whereSeparator: \.isWhitespace).joined(separator: " ")
            if let line, !line.isEmpty {
                return "title: \(line) | text: \(text)"
            }
            return "title: none | text: \(text)"
        }
    }
}

extension EmbeddingGemma2Embedding.Input.Part {
    fileprivate var text: String? {
        if case .text(let text) = self { text } else { nil }
    }
}
