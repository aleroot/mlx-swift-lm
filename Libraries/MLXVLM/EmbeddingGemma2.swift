// Copyright © 2026 Apple Inc.

import CoreImage
import Foundation
import MLX
import MLXLMCommon
import MLXNN

// MARK: - Configuration

/// Configuration of `google/embeddinggemma-2` (`model_type: embedding_gemma2`).
///
/// Decodes the checkpoint's `config.json` directly, including the nested `text_config`
/// and the optional `vision_config`. The audio encoder is never loaded.
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

    /// The vision encoder layout and the tokens that mark images in a sequence.
    public struct Vision: Sendable {
        public let encoder: Gemma4VisionConfiguration
        public let imageTokenID: Int
        public let beginImageTokenID: Int
        public let endImageTokenID: Int
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
        case imageTokenId = "image_token_id"
        case boiTokenId = "boi_token_id"
        case eoiTokenId = "eoi_token_id"
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
                encoder: encoder, imageTokenID: image, beginImageTokenID: begin,
                endImageTokenID: end)
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
/// text and image soft tokens into one normalized
/// ``EmbeddingGemma2Configuration/embeddingDim`` space. Load checkpoints with
/// ``loadWeights(modelDirectory:model:quantization:perLayerQuantization:)``.
public final class EmbeddingGemma2: Module, BaseLanguageModel {

    public let config: EmbeddingGemma2Configuration

    @ModuleInfo(key: "language_model") private var languageModel: EmbeddingGemma2TextModel
    @ModuleInfo(key: "vision_tower") private var visionTower: Gemma4VisionModel?
    @ModuleInfo(key: "embed_vision") private var embedVision: Gemma4MultimodalEmbedder?

    public init(_ config: EmbeddingGemma2Configuration) {
        self.config = config
        self._languageModel.wrappedValue = EmbeddingGemma2TextModel(config)
        if let vision = config.vision {
            self._visionTower.wrappedValue = Gemma4VisionModel(config: vision.encoder)
            self._embedVision.wrappedValue = Gemma4MultimodalEmbedder(
                embeddingDim: vision.encoder.hiddenSize, textHiddenSize: config.hiddenSize,
                eps: vision.encoder.rmsNormEps)
        }
        super.init()
    }

    /// - Returns: `[soft tokens, hidden]` features that fill one image's placeholder tokens.
    ///   The vision tower batches only images of one size.
    public func imageFeatures(_ pixels: MLXArray) -> MLXArray? {
        guard let visionTower, let embedVision else { return nil }
        return embedVision(visionTower(pixels))
    }

    /// Features of several images that may differ in size, in input order.
    ///
    /// - Returns: `[1, total soft tokens, hidden]`, or `nil` when this checkpoint has no
    ///   vision encoder.
    public func imageFeatures(_ pixels: [MLXArray]) -> MLXArray? {
        guard !pixels.isEmpty, visionTower != nil else { return nil }
        var rows: [MLXArray] = []
        rows.reserveCapacity(pixels.count)
        for item in pixels {
            guard let features = imageFeatures(item) else { return nil }
            rows.append(features.reshaped(-1, features.dim(-1)))
        }
        return concatenated(rows, axis: 0).expandedDimensions(axis: 0)
    }

    /// - Parameters:
    ///   - inputIds: `[B, L]` tokens. Image placeholders read the padding embedding until
    ///     `imageFeatures` replaces them.
    ///   - attentionMask: `[B, L]`, `1` for real tokens. Attention and pooling ignore padding.
    ///   - imageFeatures: Features of every image in `inputIds` order, `[1, soft tokens,
    ///     hidden]` or `[soft tokens, hidden]`; single row only.
    /// - Returns: `[B, embeddingDim]` unit-length float32 embeddings.
    public func embed(
        inputIds: MLXArray, attentionMask: MLXArray?, imageFeatures: MLXArray? = nil
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
        if let imageFeatures, let vision = config.vision {
            precondition(inputIds.dim(0) == 1)
            let isImage = inputIds .== Int32(vision.imageTokenID)
            let index = maximum(cumsum(isImage.asType(.int32), axis: 1) - 1, 0)
            // The tower batches: [1, soft tokens, hidden] becomes [soft tokens, hidden].
            let features = imageFeatures.asType(hidden.dtype).reshaped(-1, hidden.dim(-1))
                .take(index.squeezed(axis: 0), axis: 0).expandedDimensions(axis: 0)
            hidden = MLX.where(isImage.expandedDimensions(axis: -1), features, hidden)
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

    /// Keeps the text model and, when this checkpoint has one, the vision encoder. Drops
    /// the audio encoder, which the library does not run.
    public func sanitize(weights: [String: MLXArray]) throws -> [String: MLXArray] {
        var clean: [String: MLXArray] = [:]
        for (key, value) in weights {
            if key.hasPrefix("language_model.") {
                clean[key] = value
            } else if visionTower != nil,
                key.hasPrefix("vision_tower.")
                    || key.hasPrefix("embed_vision.")
            {
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
        let oriented = try image.asCIImage().settingProperties([
            CIImageOption.applyOrientationProperty: true
        ])
        guard !oriented.extent.isEmpty, !oriented.extent.isInfinite else {
            throw EmbeddingGemma2Embedding.Error.invalidMedia
        }
        let srgb = MediaProcessing.inSRGBToneCurveSpace(oriented)
        let target = configuration.aspectPreservingTargetSize(for: srgb.extent.size)
        let (height, width) = (Int(target.height), Int(target.width))
        var pixels = MediaProcessing.asMLXArray(srgb)
        if pixels.dim(2) != height || pixels.dim(3) != width {
            pixels = Self.resized(pixels, height: height, width: width)
        }
        return (pixels, configuration.softTokenCount(height: height, width: width))
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

// MARK: - Sequence Layout

/// Builds the token sequence of one multimodal input, as the reference processor does.
public enum EmbeddingGemma2Sequence {

    /// - Parameters:
    ///   - text: Tokens of the prepared text, starting with `<bos>` and without `<eos>`.
    ///   - imageTokenCounts: Soft tokens of each image, in input order.
    ///   - vision: The checkpoint's image token layout; required when images are present.
    /// - Returns: `text`, then `<boi> <image>×n <eoi>` per image, then `eos`. Text is
    ///   truncated to fit `limit`.
    public static func tokens(
        text: [Int], imageTokenCounts: [Int], vision: EmbeddingGemma2Configuration.Vision?,
        endOfSequence: Int, limit: Int
    ) throws -> [Int] {
        let imageLength = imageTokenCounts.reduce(0) { $0 + $1 + 2 }
        let textLimit = limit - imageLength - 1
        guard textLimit >= min(text.count, 1) else {
            throw EmbeddingGemma2Embedding.Error.contextExceeded
        }
        var tokens = Array(text.prefix(textLimit))
        if !imageTokenCounts.isEmpty {
            guard let vision else { throw EmbeddingGemma2Embedding.Error.imagesUnsupported }
            for count in imageTokenCounts {
                tokens.append(vision.beginImageTokenID)
                tokens.append(contentsOf: repeatElement(vision.imageTokenID, count: count))
                tokens.append(vision.endImageTokenID)
            }
        }
        tokens.append(endOfSequence)
        return tokens
    }
}

// MARK: - Embedding

/// Text and image embeddings from an EmbeddingGemma 2 checkpoint, in one shared space.
/// Load the checkpoint once and reuse this actor for every call.
///
/// ```swift
/// let embeddings = try await EmbeddingGemma2Embedding(
///     modelDirectory: directory, tokenizerLoader: loader)
/// let vector = try await embeddings.embed(
///     [.init(text: "What causes the northern lights?")], task: .searchQuery)[0]
/// ```
public actor EmbeddingGemma2Embedding {

    /// One independently embedded item.
    public struct Input {
        public let text: String
        /// A document title or file name; used by the `.document` task.
        public let title: String?
        public let images: [UserInput.Image]

        public init(text: String = "", title: String? = nil, images: [UserInput.Image] = []) {
            self.text = text
            self.title = title
            self.images = images
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
        case imagesUnsupported
        case invalidMedia
        case contextExceeded
        case invalidEmbedding

        public var errorDescription: String? {
            switch self {
            case .emptyInput:
                "An embedding input must contain text or images."
            case .imagesUnsupported:
                "This checkpoint does not support image embeddings."
            case .invalidMedia:
                "An image could not be decoded."
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
        // A missing processor configuration disables images; text keeps working.
        self.imageProcessor =
            config.vision == nil
            ? nil : try? EmbeddingGemma2ImageProcessor(directory: modelDirectory)
    }

    /// Embeds independent items in input order, one at a time to bound visual working
    /// memory. Use the same task for items that will be compared in the same space.
    public func embed(_ inputs: [Input], task: Task) async throws -> [[Float]] {
        try Swift.Task.checkCancellation()
        var vectors: [[Float]] = []
        vectors.reserveCapacity(inputs.count)
        for input in inputs {
            try Swift.Task.checkCancellation()
            vectors.append(try embed(input, task: task))
        }
        return vectors
    }

    /// Embeds one item; returns a unit-length float32 vector.
    public func embed(_ input: Input, task: Task) throws -> [Float] {
        let hasText = !input.text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
        guard hasText || !input.images.isEmpty else { throw Error.emptyInput }
        var images: [(pixels: MLXArray, tokens: Int)] = []
        if !input.images.isEmpty {
            guard let imageProcessor else { throw Error.imagesUnsupported }
            images = try input.images.map { try imageProcessor.pixels(for: $0) }
        }

        // Task prefixes apply to text only; media without text is embedded as-is.
        let prepared = hasText ? Self.prompt(text: input.text, title: input.title, task: task) : ""
        guard let endOfSequence = tokenizer.eosTokenId ?? tokenizer.convertTokenToId("<eos>") else {
            throw Error.invalidEmbedding
        }
        var textTokens = tokenizer.encode(text: prepared, addSpecialTokens: true)
        // The checkpoint's tokenizer configuration omits `add_eos_token`, so the
        // tokenizer drops the `<eos>` its post-processor declares and the model expects.
        if textTokens.last == endOfSequence { textTokens.removeLast() }
        let tokens = try EmbeddingGemma2Sequence.tokens(
            text: textTokens, imageTokenCounts: images.map(\.tokens),
            vision: model.config.vision, endOfSequence: endOfSequence,
            limit: EmbeddingGemma2Configuration.contextLength)

        // Each image keeps its own size. The vision tower only batches equal sizes.
        let features: MLXArray?
        if images.isEmpty {
            features = nil
        } else if let gathered = model.imageFeatures(images.map(\.pixels)) {
            features = gathered
        } else {
            throw Error.imagesUnsupported
        }
        let vector = model.embed(
            inputIds: MLXArray(tokens.map(Int32.init), [1, tokens.count]),
            attentionMask: nil, imageFeatures: features)
        try MLX.checkedEval(vector)
        try Swift.Task.checkCancellation()
        let values = vector.asArray(Float.self)
        guard values.allSatisfy(\.isFinite), values.contains(where: { $0 != 0 }) else {
            throw Error.invalidEmbedding
        }
        return values
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
