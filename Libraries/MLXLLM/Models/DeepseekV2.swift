// Copyright © 2026 Apple Inc.

// port of https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/deepseek_v2.py

import Foundation
import MLX
import MLXLMCommon
import MLXNN

public struct DeepseekV2Configuration: Codable, Sendable {
    var vocabSize: Int
    var hiddenSize: Int
    var intermediateSize: Int
    var moeIntermediateSize: Int
    var numHiddenLayers: Int
    var numAttentionHeads: Int
    var numKeyValueHeads: Int
    var nSharedExperts: Int?
    var nRoutedExperts: Int?
    var routedScalingFactor: Float
    var kvLoraRank: Int
    var qLoraRank: Int?
    var qkRopeHeadDim: Int
    var vHeadDim: Int
    var qkNopeHeadDim: Int
    var topkMethod: String
    var nGroup: Int?
    var topkGroup: Int?
    var numExpertsPerTok: Int?
    var moeLayerFreq: Int
    var firstKDenseReplace: Int
    var maxPositionEmbeddings: Int
    var rmsNormEps: Float
    var ropeTheta: Float
    var ropeScaling: [String: StringOrNumber]?
    var attentionBias: Bool

    enum CodingKeys: String, CodingKey {
        case vocabSize = "vocab_size"
        case hiddenSize = "hidden_size"
        case intermediateSize = "intermediate_size"
        case moeIntermediateSize = "moe_intermediate_size"
        case numHiddenLayers = "num_hidden_layers"
        case numAttentionHeads = "num_attention_heads"
        case numKeyValueHeads = "num_key_value_heads"
        case nSharedExperts = "n_shared_experts"
        case nRoutedExperts = "n_routed_experts"
        case routedScalingFactor = "routed_scaling_factor"
        case kvLoraRank = "kv_lora_rank"
        case qLoraRank = "q_lora_rank"
        case qkRopeHeadDim = "qk_rope_head_dim"
        case vHeadDim = "v_head_dim"
        case qkNopeHeadDim = "qk_nope_head_dim"
        case topkMethod = "topk_method"
        case nGroup = "n_group"
        case topkGroup = "topk_group"
        case numExpertsPerTok = "num_experts_per_tok"
        case moeLayerFreq = "moe_layer_freq"
        case firstKDenseReplace = "first_k_dense_replace"
        case maxPositionEmbeddings = "max_position_embeddings"
        case rmsNormEps = "rms_norm_eps"
        case ropeTheta = "rope_theta"
        case ropeScaling = "rope_scaling"
        case attentionBias = "attention_bias"
    }
}

class DeepseekV2Attention: Module {
    let numHeads: Int
    let maxPositionEmbeddings: Int
    let ropeTheta: Float
    let qLoraRank: Int?
    let qkRopeHeadDim: Int
    let kvLoraRank: Int
    let vHeadDim: Int
    let qkNopeHeadDim: Int
    let qHeadDim: Int
    var scale: Float

    let rope: RoPELayer
    @ModuleInfo(key: "q_proj") var qProj: Linear?
    @ModuleInfo(key: "q_a_proj") var qAProj: Linear?
    @ModuleInfo(key: "q_a_layernorm") var qALayerNorm: RMSNorm?
    @ModuleInfo(key: "q_b_proj") var qBProj: Linear?
    @ModuleInfo(key: "o_proj") var oProj: Linear
    @ModuleInfo(key: "kv_a_proj_with_mqa") var kvAProjWithMqa: Linear
    @ModuleInfo(key: "kv_a_layernorm") var kvALayerNorm: RMSNorm
    @ModuleInfo(key: "kv_b_proj") var kvBProj: Linear

    init(config: DeepseekV2Configuration) {
        self.numHeads = config.numAttentionHeads
        self.maxPositionEmbeddings = config.maxPositionEmbeddings
        self.ropeTheta = config.ropeTheta
        self.qLoraRank = config.qLoraRank
        self.qkRopeHeadDim = config.qkRopeHeadDim
        self.kvLoraRank = config.kvLoraRank
        self.vHeadDim = config.vHeadDim
        self.qkNopeHeadDim = config.qkNopeHeadDim
        self.qHeadDim = config.qkNopeHeadDim + config.qkRopeHeadDim
        self.scale = pow(Float(qHeadDim), -0.5)

        if let qLoraRank {
            self._qAProj.wrappedValue = Linear(
                config.hiddenSize, qLoraRank, bias: config.attentionBias)
            self._qALayerNorm.wrappedValue = RMSNorm(dimensions: qLoraRank)
            self._qBProj.wrappedValue = Linear(
                qLoraRank, numHeads * qHeadDim, bias: false)
        } else {
            self._qProj.wrappedValue = Linear(
                config.hiddenSize, numHeads * qHeadDim, bias: false)
        }

        self._kvAProjWithMqa.wrappedValue = Linear(
            config.hiddenSize, kvLoraRank + qkRopeHeadDim, bias: config.attentionBias)
        self._kvALayerNorm.wrappedValue = RMSNorm(dimensions: kvLoraRank)
        self._kvBProj.wrappedValue = Linear(
            kvLoraRank,
            numHeads * (qHeadDim - qkRopeHeadDim + vHeadDim),
            bias: false)
        self._oProj.wrappedValue = Linear(
            numHeads * vHeadDim, config.hiddenSize, bias: config.attentionBias)

        if let ropeScaling = config.ropeScaling {
            let mScaleAllDim = ropeScaling["mscale_all_dim"]?.asFloat() ?? 0
            if mScaleAllDim != 0 {
                let scalingFactor = ropeScaling["factor"]?.asFloat() ?? 1
                if scalingFactor > 1 {
                    let mscale = 0.1 * mScaleAllDim * log(scalingFactor) + 1
                    self.scale = self.scale * mscale * mscale
                }
            }
        }

        self.rope = initializeRope(
            dims: qkRopeHeadDim,
            base: ropeTheta,
            traditional: true,
            scalingConfig: config.ropeScaling,
            maxPositionEmbeddings: maxPositionEmbeddings
        )
    }

    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        let (B, L, _) = (x.dim(0), x.dim(1), x.dim(2))

        let q: MLXArray
        if qLoraRank == nil {
            q = qProj!(x)
        } else {
            q = qBProj!(qALayerNorm!(qAProj!(x)))
        }

        let reshapedQ = q.reshaped(B, L, numHeads, qHeadDim).transposed(0, 2, 1, 3)
        let splitQ = split(reshapedQ, indices: [qkNopeHeadDim], axis: -1)
        let qNope = splitQ[0]
        var qPe = splitQ[1]

        let compressedKv = kvAProjWithMqa(x)
        let splitCompressedKv = split(compressedKv, indices: [kvLoraRank], axis: -1)
        let compressedKvStates = splitCompressedKv[0]
        var kPe = splitCompressedKv[1]
        kPe = kPe.reshaped(B, L, 1, qkRopeHeadDim).transposed(0, 2, 1, 3)

        var kv = kvBProj(kvALayerNorm(compressedKvStates))
        kv = kv.reshaped(B, L, numHeads, -1).transposed(0, 2, 1, 3)
        let splitKv = split(kv, indices: [qkNopeHeadDim], axis: -1)
        let kNope = splitKv[0]
        var values = splitKv[1]

        qPe = applyRotaryPosition(rope, to: qPe, cache: cache)
        kPe = applyRotaryPosition(rope, to: kPe, cache: cache)
        kPe = repeated(kPe, count: numHeads, axis: 1)

        let keys: MLXArray
        if let cache {
            (keys, values) = cache.update(
                keys: concatenated([kNope, kPe], axis: -1), values: values)
        } else {
            keys = concatenated([kNope, kPe], axis: -1)
        }

        let queries = concatenated([qNope, qPe], axis: -1)

        let output = attentionWithCacheUpdate(
            queries: queries,
            keys: keys,
            values: values,
            cache: cache,
            scale: scale,
            mask: mask
        )
        .transposed(0, 2, 1, 3)
        .reshaped(B, L, -1)

        return oProj(output)
    }
}

class DeepseekV2MLP: Module, UnaryLayer {
    @ModuleInfo(key: "gate_proj") var gateProj: Linear
    @ModuleInfo(key: "up_proj") var upProj: Linear
    @ModuleInfo(key: "down_proj") var downProj: Linear

    init(
        config: DeepseekV2Configuration, hiddenSize: Int? = nil, intermediateSize: Int? = nil
    ) {
        let hiddenSize = hiddenSize ?? config.hiddenSize
        let intermediateSize = intermediateSize ?? config.intermediateSize
        self._gateProj.wrappedValue = Linear(hiddenSize, intermediateSize, bias: false)
        self._upProj.wrappedValue = Linear(hiddenSize, intermediateSize, bias: false)
        self._downProj.wrappedValue = Linear(intermediateSize, hiddenSize, bias: false)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        downProj(silu(gateProj(x)) * upProj(x))
    }
}

class DeepseekV2MoEGate: Module {
    let topK: Int
    let nRoutedExperts: Int
    let routedScalingFactor: Float
    let topkMethod: String
    let nGroup: Int?
    let topkGroup: Int?

    @ParameterInfo(key: "weight") var weight: MLXArray

    init(config: DeepseekV2Configuration) {
        self.topK = config.numExpertsPerTok ?? 1
        self.nRoutedExperts = config.nRoutedExperts ?? 1
        self.routedScalingFactor = config.routedScalingFactor
        self.topkMethod = config.topkMethod
        self.nGroup = config.nGroup
        self.topkGroup = config.topkGroup
        self._weight.wrappedValue = zeros([nRoutedExperts, config.hiddenSize])
    }

    func callAsFunction(_ x: MLXArray) -> (MLXArray, MLXArray) {
        let gates = x.matmul(weight.T)
        var scores = softmax(gates.asType(.float32), axis: -1, precise: true)

        if topkMethod == "group_limited_greedy",
            let nGroup,
            let topkGroup,
            nGroup > topkGroup
        {
            let (bsz, seqLen, _) = (x.dim(0), x.dim(1), x.dim(2))
            scores = scores.reshaped(bsz, seqLen, nGroup, -1)
            let groupScores = scores.max(axis: -1, keepDims: true)
            let k = nGroup - topkGroup
            let groupIdx = argPartition(groupScores, kth: k - 1, axis: -2)[
                .ellipsis, ..<k, 0...
            ]
            scores = putAlong(scores, groupIdx, values: MLXArray(0.0), axis: -2)
            scores = flattened(scores, start: -2, end: -1)
        }

        let inds = argPartition(-scores, kth: topK - 1, axis: -1)[.ellipsis, ..<topK]
        let selectedScores = takeAlong(scores, inds, axis: -1) * routedScalingFactor
        return (inds, selectedScores)
    }
}

class DeepseekV2MoE: Module, UnaryLayer {
    @ModuleInfo(key: "switch_mlp") var switchMLP: SwitchGLU
    @ModuleInfo(key: "gate") var gate: DeepseekV2MoEGate
    @ModuleInfo(key: "shared_experts") var sharedExperts: DeepseekV2MLP?

    init(config: DeepseekV2Configuration) {
        self._switchMLP.wrappedValue = SwitchGLU(
            inputDims: config.hiddenSize,
            hiddenDims: config.moeIntermediateSize,
            numExperts: config.nRoutedExperts ?? 1
        )
        self._gate.wrappedValue = DeepseekV2MoEGate(config: config)

        if let sharedExpertCount = config.nSharedExperts {
            self._sharedExperts.wrappedValue = DeepseekV2MLP(
                config: config,
                intermediateSize: config.moeIntermediateSize * sharedExpertCount
            )
        }
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let (inds, scores) = gate(x)
        var y = switchMLP(x, inds)
        y = (y * scores[.ellipsis, .newAxis]).sum(axis: -2)
        if let sharedExperts {
            y = y + sharedExperts(x)
        }
        return y
    }
}

class DeepseekV2DecoderLayer: Module {
    @ModuleInfo(key: "self_attn") var selfAttn: DeepseekV2Attention
    var mlp: UnaryLayer
    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm

    init(config: DeepseekV2Configuration, layerIdx: Int) {
        self._selfAttn.wrappedValue = DeepseekV2Attention(config: config)
        if config.nRoutedExperts != nil,
            layerIdx >= config.firstKDenseReplace,
            layerIdx % config.moeLayerFreq == 0
        {
            self.mlp = DeepseekV2MoE(config: config)
        } else {
            self.mlp = DeepseekV2MLP(config: config)
        }

        self._inputLayerNorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
        self._postAttentionLayerNorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
    }

    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        let r = selfAttn(inputLayerNorm(x), mask: mask, cache: cache)
        let h = x + r
        let r2 = mlp(postAttentionLayerNorm(h))
        return h + r2
    }
}

public class DeepseekV2ModelInner: Module {
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    var layers: [DeepseekV2DecoderLayer]
    @ModuleInfo(key: "norm") var norm: RMSNorm

    init(config: DeepseekV2Configuration) {
        self._embedTokens.wrappedValue = Embedding(
            embeddingCount: config.vocabSize, dimensions: config.hiddenSize)
        self.layers = (0 ..< config.numHiddenLayers).map {
            DeepseekV2DecoderLayer(config: config, layerIdx: $0)
        }
        self._norm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
    }

    func callAsFunction(_ x: MLXArray, cache: [KVCache]?) -> MLXArray {
        var h = embedTokens(x)
        let attentionMask = createAttentionMask(h: h, cache: cache?.first)

        for (i, layer) in layers.enumerated() {
            h = layer(h, mask: attentionMask, cache: cache?[i])
        }

        return norm(h)
    }
}

public class DeepseekV2Model: Module, LLMModel, KVCacheDimensionProvider, LoRAModel {
    public let kvHeads: [Int]

    let args: DeepseekV2Configuration
    public let model: DeepseekV2ModelInner
    @ModuleInfo(key: "lm_head") var lmHead: Linear

    init(_ args: DeepseekV2Configuration) {
        self.args = args
        self.kvHeads = Array(repeating: args.numKeyValueHeads, count: args.numHiddenLayers)
        self.model = DeepseekV2ModelInner(config: args)
        self._lmHead.wrappedValue = Linear(args.hiddenSize, args.vocabSize, bias: false)
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]? = nil) -> MLXArray {
        let out = model(inputs, cache: cache)
        return lmHead(out)
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var sanitized = weights

        for layer in 0 ..< args.numHiddenLayers {
            let prefix = "model.layers.\(layer)"
            for projName in ["gate_proj", "down_proj", "up_proj"] {
                for key in ["weight", "scales", "biases"] {
                    let firstKey = "\(prefix).mlp.experts.0.\(projName).\(key)"
                    guard sanitized[firstKey] != nil else { continue }

                    var stackedWeights: [MLXArray] = []
                    for expert in 0 ..< (args.nRoutedExperts ?? 0) {
                        let expertKey = "\(prefix).mlp.experts.\(expert).\(projName).\(key)"
                        if let value = sanitized.removeValue(forKey: expertKey) {
                            stackedWeights.append(value)
                        }
                    }

                    if !stackedWeights.isEmpty {
                        sanitized["\(prefix).mlp.switch_mlp.\(projName).\(key)"] = stacked(
                            stackedWeights)
                    }
                }
            }
        }

        return sanitized
    }

    public var loraLayers: [Module] {
        model.layers
    }
}
