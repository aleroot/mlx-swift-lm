// Copyright © 2026 Apple Inc.

import CoreMedia
import Foundation
import MLX
import MLXLMCommon
import MLXNN
import Testing

@testable import MLXVLM

/// Deterministic-checkpoint tests for ``EmbeddingGemma2``: reference vectors and audio
/// features from the Transformers float32 implementation, the checkpoint tensor manifest,
/// sequence layout, frame sampling, image resize weights, and task prompts. No weights are
/// downloaded.
struct EmbeddingGemma2Tests {

    /// A tiny text-only config with sliding and global layers of different head sizes.
    private static let tinyConfig =
        #"{"text_config":{"hidden_size":16,"num_hidden_layers":4,"num_attention_heads":2,"num_key_value_heads":2,"head_dim":4,"intermediate_size":32,"hidden_size_per_layer_input":8,"embedding_dim":12,"vocab_size":32,"rms_norm_eps":1e-6,"sliding_window":2,"layer_types":["sliding_attention","full_attention","sliding_attention","full_attention"],"per_layer_config":{"01":{"head_dim":8,"num_key_value_heads":1},"03":{"head_dim":8,"num_key_value_heads":1}},"rope_parameters":{"full_attention":{"rope_theta":1000000.0,"rope_type":"default"},"sliding_attention":{"rope_theta":10000.0,"rope_type":"default"}}}}"#

    /// The tiny config plus a small vision encoder and image tokens.
    private static let tinyVisionConfig =
        #"{"text_config":{"hidden_size":16,"num_hidden_layers":2,"num_attention_heads":2,"num_key_value_heads":1,"head_dim":8,"intermediate_size":32,"hidden_size_per_layer_input":8,"embedding_dim":12,"vocab_size":32,"rms_norm_eps":1e-06,"sliding_window":4,"layer_types":["sliding_attention","full_attention"],"per_layer_config":{"01":{"head_dim":8,"num_key_value_heads":1}},"rope_parameters":{"full_attention":{"rope_theta":1000000.0,"rope_type":"default"},"sliding_attention":{"rope_theta":10000.0,"rope_type":"default"}}},"vision_config":{"model_type":"gemma4_vision","hidden_size":32,"intermediate_size":64,"num_hidden_layers":2,"num_attention_heads":2,"num_key_value_heads":2,"head_dim":16,"patch_size":4,"pooling_kernel_size":2,"position_embedding_size":64,"default_output_length":9,"rms_norm_eps":1e-06,"use_clipped_linears":false,"standardize":false,"rope_parameters":{"rope_theta":100.0,"rope_type":"axial"}},"image_token_id":7,"boi_token_id":5,"eoi_token_id":6}"#

    /// `text_config`, `vision_config`, `audio_config`, and media tokens of
    /// `google/embeddinggemma-2`.
    private static let checkpointConfig =
        #"{"text_config":{"attention_bias":false,"attention_dropout":0.0,"bos_token_id":2,"dtype":"bfloat16","embedding_dim":768,"eos_token_id":1,"head_dim":256,"hidden_activation":"gelu_pytorch_tanh","hidden_size":512,"hidden_size_per_layer_input":512,"initializer_range":0.02,"intermediate_size":2048,"layer_types":["sliding_attention","sliding_attention","sliding_attention","sliding_attention","sliding_attention","full_attention","sliding_attention","sliding_attention","sliding_attention","sliding_attention","sliding_attention","full_attention","sliding_attention","sliding_attention","sliding_attention","sliding_attention","sliding_attention","full_attention","sliding_attention","sliding_attention","sliding_attention","sliding_attention","sliding_attention","full_attention"],"max_position_embeddings":262144,"model_type":"embedding_gemma2_text","num_attention_heads":4,"num_hidden_layers":24,"num_key_value_heads":2,"pad_token_id":0,"per_layer_config":{"05":{"head_dim":512,"num_key_value_heads":1},"11":{"head_dim":512,"num_key_value_heads":1},"17":{"head_dim":512,"num_key_value_heads":1},"23":{"head_dim":512,"num_key_value_heads":1}},"rms_norm_eps":1e-06,"rope_parameters":{"full_attention":{"rope_theta":1000000.0,"rope_type":"default"},"sliding_attention":{"rope_theta":10000.0,"rope_type":"default"}},"sliding_window":512,"vocab_size":262144},"vision_config":{"model_type":"gemma4_vision","hidden_size":768,"intermediate_size":3072,"num_hidden_layers":16,"num_attention_heads":12,"num_key_value_heads":12,"head_dim":64,"patch_size":16,"pooling_kernel_size":3,"position_embedding_size":10240,"default_output_length":280,"rms_norm_eps":1e-06,"use_clipped_linears":false,"standardize":false,"rope_parameters":{"rope_theta":100.0,"rope_type":"axial"}},"audio_config":{"attention_chunk_size":12,"attention_context_left":13,"attention_context_right":0,"attention_invalid_logits_value":-1000000000.0,"attention_logit_cap":50.0,"conv_kernel_size":5,"gradient_clipping":10000000000.0,"hidden_act":"silu","hidden_size":1024,"model_type":"gemma4_audio","num_attention_heads":8,"num_hidden_layers":12,"output_proj_dims":1536,"residual_weight":0.5,"rms_norm_eps":1e-06,"subsampling_conv_channels":[128,32],"use_clipped_linears":true},"image_token_id":258880,"video_token_id":258884,"boi_token_id":255999,"eoi_token_id":258882,"audio_token_id":258881,"boa_token_id":256000,"eoa_token_index":258883}"#

    /// The tiny vision config plus a tiny audio encoder and video and audio tokens.
    private static let tinyMediaConfig = tinyVisionConfig.replacingOccurrences(
        of: #""eoi_token_id":6}"#,
        with:
            #""eoi_token_id":6,"video_token_id":8,"audio_token_id":9,"boa_token_id":10,"eoa_token_index":11,"audio_config":"#
            + tinyAudioConfig(futureContext: 0) + "}")

    private static func decode(_ json: String) throws -> EmbeddingGemma2Configuration {
        try JSONDecoder().decode(EmbeddingGemma2Configuration.self, from: Data(json.utf8))
    }

    private static func deterministicWeights(for model: Module) throws -> [String: MLXArray] {
        Dictionary(
            uniqueKeysWithValues: model.parameters().flattened().map { (name, value) in
                (name, MLXArray((0 ..< value.size).map { Float($0 % 17 - 8) * 0.05 }, value.shape))
            })
    }

    @Test
    func checkpointConfigurationResolvesAttentionAndEncoders() throws {
        let config = try Self.decode(Self.checkpointConfig)
        let sliding = EmbeddingGemma2Configuration.Attention(
            headDim: 256, kvHeads: 2, ropeBase: 10_000, isGlobal: false)
        let global = EmbeddingGemma2Configuration.Attention(
            headDim: 512, kvHeads: 1, ropeBase: 1_000_000, isGlobal: true)
        #expect(
            config.attention == (0 ..< 24).map { [5, 11, 17, 23].contains($0) ? global : sliding })
        #expect(config.slidingWindow == 512)
        #expect(config.embeddingDim == 768)
        let vision = try #require(config.vision)
        #expect(vision.imageTokenID == 258_880)
        #expect(vision.beginImageTokenID == 255_999)
        #expect(vision.endImageTokenID == 258_882)
        #expect(vision.videoTokenID == 258_884)
        #expect(vision.encoder.hiddenSize == 768)
        let audio = try #require(config.audio)
        #expect(audio.audioTokenID == 258_881)
        #expect(audio.beginAudioTokenID == 256_000)
        #expect(audio.endAudioTokenID == 258_883)
        #expect(audio.encoder.outputProjectionDimensions == 1536)
    }

    @Test
    func perLayerKeysMustBeLayerIndices() {
        let invalid = Self.tinyConfig.replacingOccurrences(of: #""03":"#, with: #""last":"#)
        #expect(throws: DecodingError.self) { try Self.decode(invalid) }
    }

    @Test
    func clippedOrStandardizedVisionEncodersStayTextOnly() throws {
        for flag in ["use_clipped_linears", "standardize"] {
            let json = Self.tinyVisionConfig.replacingOccurrences(
                of: #""\#(flag)":false"#, with: #""\#(flag)":true"#)
            #expect(try Self.decode(json).vision == nil)
        }
    }

    /// Names and shapes of every tensor in the `mlx-community/embeddinggemma-2-bf16`
    /// safetensors header, through the module tree the loader updates.
    @Test
    func modulesMatchTheCheckpointTensors() throws {
        let model = EmbeddingGemma2(try Self.decode(Self.checkpointConfig))
        var expected: [String: [Int]] = [
            "language_model.embed_tokens.weight": [262144, 512],
            "language_model.embedding_projection.weight": [768, 512],
            "language_model.norm.weight": [512],
            "language_model.ple.per_layer_model_projection.weight": [12288, 512],
            "language_model.ple.per_layer_projection_norm.weight": [512],
            "embed_vision.embedding_projection.weight": [512, 768],
            "vision_tower.patch_embedder.input_proj.weight": [768, 768],
            "vision_tower.patch_embedder.position_embedding_table": [2, 10240, 768],
            "embed_audio.embedding_projection.weight": [512, 1536],
            "audio_tower.output_proj.weight": [1536, 1024],
            "audio_tower.output_proj.bias": [1536],
            "audio_tower.subsample_conv_projection.layer0.conv.weight": [128, 3, 3, 1],
            "audio_tower.subsample_conv_projection.layer0.norm.weight": [128],
            "audio_tower.subsample_conv_projection.layer1.conv.weight": [32, 3, 3, 128],
            "audio_tower.subsample_conv_projection.layer1.norm.weight": [32],
            "audio_tower.subsample_conv_projection.input_proj_linear.weight": [1024, 1024],
        ]
        for layer in 0 ..< 24 {
            let width = [5, 11, 17, 23].contains(layer) ? 512 : 256
            let tensors: [String: [Int]] = [
                "input_layernorm.weight": [512], "layer_scalar": [1],
                "mlp.down_proj.weight": [512, 2048], "mlp.gate_proj.weight": [2048, 512],
                "mlp.up_proj.weight": [2048, 512],
                "ple_block.per_layer_input_gate.weight": [512, 512],
                "ple_block.per_layer_projection.weight": [512, 512],
                "ple_block.post_per_layer_input_norm.weight": [512],
                "post_attention_layernorm.weight": [512],
                "post_feedforward_layernorm.weight": [512],
                "pre_feedforward_layernorm.weight": [512],
                "self_attn.q_norm.weight": [width], "self_attn.k_norm.weight": [width],
                "self_attn.q_proj.weight": [4 * width, 512],
                "self_attn.o_proj.weight": [512, 4 * width],
                "self_attn.k_proj.weight": [512, 512], "self_attn.v_proj.weight": [512, 512],
            ]
            for (name, shape) in tensors {
                expected["language_model.layers.\(layer).\(name)"] = shape
            }
        }
        for layer in 0 ..< 16 {
            let tensors: [String: [Int]] = [
                "input_layernorm.weight": [768], "post_attention_layernorm.weight": [768],
                "pre_feedforward_layernorm.weight": [768],
                "post_feedforward_layernorm.weight": [768],
                "mlp.gate_proj.linear.weight": [3072, 768],
                "mlp.up_proj.linear.weight": [3072, 768],
                "mlp.down_proj.linear.weight": [768, 3072],
                "self_attn.q_proj.linear.weight": [768, 768],
                "self_attn.k_proj.linear.weight": [768, 768],
                "self_attn.v_proj.linear.weight": [768, 768],
                "self_attn.o_proj.linear.weight": [768, 768],
                "self_attn.q_norm.weight": [64], "self_attn.k_norm.weight": [64],
            ]
            for (name, shape) in tensors {
                expected["vision_tower.encoder.layers.\(layer).\(name)"] = shape
            }
        }
        for layer in 0 ..< 12 {
            var tensors: [String: [Int]] = [
                "feed_forward1.pre_layer_norm.weight": [1024],
                "feed_forward1.post_layer_norm.weight": [1024],
                "feed_forward2.pre_layer_norm.weight": [1024],
                "feed_forward2.post_layer_norm.weight": [1024],
                "lconv1d.depthwise_conv1d.weight": [1024, 5, 1],
                "lconv1d.pre_layer_norm.weight": [1024], "lconv1d.conv_norm.weight": [1024],
                "self_attn.relative_k_proj.weight": [1024, 1024],
                "self_attn.per_dim_scale": [128],
                "norm_pre_attn.weight": [1024], "norm_post_attn.weight": [1024],
                "norm_out.weight": [1024],
            ]
            let clipped: [(String, [Int])] = [
                ("feed_forward1.ffw_layer_1", [4096, 1024]),
                ("feed_forward1.ffw_layer_2", [1024, 4096]),
                ("feed_forward2.ffw_layer_1", [4096, 1024]),
                ("feed_forward2.ffw_layer_2", [1024, 4096]),
                ("lconv1d.linear_start", [2048, 1024]), ("lconv1d.linear_end", [1024, 1024]),
                ("self_attn.q_proj", [1024, 1024]), ("self_attn.k_proj", [1024, 1024]),
                ("self_attn.v_proj", [1024, 1024]), ("self_attn.post", [1024, 1024]),
            ]
            for (name, shape) in clipped {
                tensors["\(name).linear.weight"] = shape
                for bound in ["input_min", "input_max", "output_min", "output_max"] {
                    tensors["\(name).\(bound)"] = []
                }
            }
            for (name, shape) in tensors {
                expected["audio_tower.layers.\(layer).\(name)"] = shape
            }
        }
        let actual = Dictionary(
            uniqueKeysWithValues: model.parameters().flattened().map { (name, value) in
                (name, value.shape)
            })
        #expect(actual.count == 413 + 211 + 752)
        #expect(actual == expected)
    }

    @Test(arguments: [true, false])
    func sanitizeKeepsTheConfiguredEncoders(_ hasMedia: Bool) throws {
        let tensor = MLXArray.zeros([1])
        let model = EmbeddingGemma2(
            try Self.decode(hasMedia ? Self.tinyMediaConfig : Self.tinyConfig))
        let clean = try model.sanitize(
            weights: [
                "language_model.norm.weight": tensor,
                "vision_tower.patch_embedder.input_proj.weight": tensor,
                "embed_vision.embedding_projection.weight": tensor,
                "audio_tower.output_proj.weight": tensor,
                "embed_audio.embedding_projection.weight": tensor,
            ])
        let media: Set<String> = [
            "vision_tower.patch_embedder.input_proj.weight",
            "embed_vision.embedding_projection.weight",
            "audio_tower.output_proj.weight",
            "embed_audio.embedding_projection.weight",
        ]
        #expect(
            Set(clean.keys)
                == (hasMedia
                    ? media.union(["language_model.norm.weight"]) : ["language_model.norm.weight"])
        )
    }

    /// PyTorch checkpoints store convolution kernels channels-first.
    @Test
    func sanitizeMovesConvolutionChannelsLast() throws {
        let model = EmbeddingGemma2(try Self.decode(Self.tinyMediaConfig))
        let kernels = [
            "audio_tower.subsample_conv_projection.layer1.conv.weight": [4, 8, 3, 3],
            "audio_tower.layers.0.lconv1d.depthwise_conv1d.weight": [32, 1, 3],
        ]
        let clean = try model.sanitize(
            weights: kernels.mapValues { MLXArray(Array(0 ..< Int32($0.reduce(1, *))), $0) })
        let layer1 = try #require(clean["audio_tower.subsample_conv_projection.layer1.conv.weight"])
        #expect(layer1.shape == [4, 3, 3, 8])
        #expect(layer1[1, 2, 0, 5].item(Int32.self) == ((1 * 8 + 5) * 3 + 2) * 3)
        let depthwise = try #require(clean["audio_tower.layers.0.lconv1d.depthwise_conv1d.weight"])
        #expect(depthwise.shape == [32, 3, 1])
        #expect(depthwise[7, 2, 0].item(Int32.self) == 7 * 3 + 2)
    }

    /// Media keeps its place among the text: `<boi> <image|video>×n <eoi>` per image and
    /// video frame, `<boa> <audio>×n <eoa>` per audio.
    @Test
    func sequenceInterleavesMediaAndTruncatesText() throws {
        let config = try Self.decode(Self.tinyMediaConfig)
        func tokens(_ segments: [EmbeddingGemma2Sequence.Segment], limit: Int = 64) throws -> [Int]
        {
            try EmbeddingGemma2Sequence.tokens(
                segments, configuration: config, beginOfSequence: 2, endOfSequence: 1,
                limit: limit)
        }
        #expect(
            try tokens([
                .text([20, 21]), .image(tokens: 2), .text([22]),
                .video(frames: 2, tokensPerFrame: 1), .audio(tokens: 3),
            ]) == [2, 20, 21, 5, 7, 7, 6, 22, 5, 8, 6, 5, 8, 6, 10, 9, 9, 9, 11, 1])
        #expect(try tokens([.audio(tokens: 0)]) == [2, 10, 11, 1])
        #expect(
            try tokens([.text([20, 21, 22]), .image(tokens: 1), .text([23, 24])], limit: 9)
                == [2, 20, 21, 22, 5, 7, 6, 23, 1])
        #expect(throws: EmbeddingGemma2Embedding.Error.contextExceeded) {
            try tokens([.text([20]), .image(tokens: 5)], limit: 8)
        }
        #expect(try tokens([.text([20, 7])]) == [2, 20, 7, 1])
        #expect(throws: EmbeddingGemma2Embedding.Error.placeholderInText) {
            try tokens([.text([20, 7]), .image(tokens: 1)])
        }
        let visionOnly = try Self.decode(Self.tinyVisionConfig)
        #expect(throws: EmbeddingGemma2Embedding.Error.unsupportedMedia) {
            try EmbeddingGemma2Sequence.tokens(
                [.audio(tokens: 1)], configuration: visionOnly, beginOfSequence: 2,
                endOfSequence: 1, limit: 64)
        }
    }

    /// Expected weights are `torch.nn.functional.interpolate(mode: "bicubic", antialias: true)`
    /// applied to unit impulses.
    @Test
    func resizeWeightsMatchAntialiasedBicubic() {
        let downscale: [Float] = [
            0.351812, 0.421268, 0.242666, 0.029551, -0.031708, -0.013589, 0, 0, 0, 0, -0.029034,
            0.027059, 0.2222, 0.38574, 0.322141, 0.114359, -0.015998, -0.024689, -0.001778, 0, 0,
            -0.001778, -0.024689, -0.015998, 0.114359, 0.322141, 0.38574, 0.2222, 0.027059,
            -0.029034, 0, 0, 0, 0, -0.013589, -0.031708, 0.029551, 0.242666, 0.421268, 0.351812,
        ]
        let upscale: [Float] = [
            1.079327, -0.079327, 0, 0, 0.697947, 0.340234, -0.038181, 0, 0.045264, 0.985457,
            -0.030722, 0, -0.0625, 0.5625, 0.5625, -0.0625, 0, -0.030722, 0.985457, 0.045265, 0,
            -0.038181, 0.340234, 0.697947, 0, 0, -0.079327, 1.079327,
        ]
        for (input, output, expected) in [(10, 4, downscale), (4, 7, upscale)] {
            let weights = EmbeddingGemma2ImageProcessor.bicubicWeights(input: input, output: output)
            #expect(weights.shape == [output, input])
            #expect(
                zip(weights.asArray(Float.self), expected).allSatisfy { abs($0 - $1) < 0.000002 })
        }
    }

    /// Expected rows come from the Transformers float32 implementation with the same
    /// deterministic weights, followed by masked mean pooling and L2 normalization.
    @Test
    func matchesTransformersReferenceWithSlidingWindowAndPadding() throws {
        let model = EmbeddingGemma2(try Self.decode(Self.tinyConfig))
        try model.update(
            parameters: ModuleParameters.unflattened(try Self.deterministicWeights(for: model)),
            verify: .all)
        let ids = MLXArray(
            Array(1 ..< 10).map(Int32.init) + Array(3 ..< 9).map(Int32.init) + [0, 0, 0], [2, 9])
        let mask = MLXArray(Array(repeating: Int32(1), count: 15) + [0, 0, 0], [2, 9])
        let expected: [[Float]] = [
            [
                -0.062816948, 0.280682176, 0.407530546, 0.737360120, 0.214800760, 0.127999231,
                0.049736522, 0.164332777, 0.185940713, 0.187425271, 0.166393027, 0.134533703,
            ],
            [
                -0.554485321, -0.071141891, -0.274588585, 0.334987015, -0.038480520, 0.090331979,
                0.037629712, 0.189092800, 0.249697328, 0.304722756, 0.351788193, 0.417249590,
            ],
        ]

        let batched = model.embed(inputIds: ids, attentionMask: mask)
        try MLX.checkedEval(batched)
        #expect(batched.shape == [2, 12])
        #expect(
            zip(batched.asArray(Float.self), expected.flatMap { $0 }).allSatisfy {
                abs($0 - $1) < 0.00001
            })

        let single = model.embed(inputIds: ids[1..., 0 ..< 6], attentionMask: mask[1..., 0 ..< 6])
        try MLX.checkedEval(single)
        #expect(zip(single.asArray(Float.self), expected[1]).allSatisfy { abs($0 - $1) < 0.00001 })
    }

    /// Expected row comes from the Transformers float32 implementation with the same
    /// deterministic weights: a 32×16 image with 32 real patches inside a 36-patch padded
    /// budget, among text tokens.
    @Test
    func matchesTransformersReferenceWithAnImage() throws {
        let config = try Self.decode(Self.tinyVisionConfig)
        let model = EmbeddingGemma2(config)
        try model.update(
            parameters: ModuleParameters.unflattened(try Self.deterministicWeights(for: model)),
            verify: .all)
        let values: [Float] = (0 ..< (3 * 16 * 32)).map { index in
            let (c, y, x) = (index / 512, index / 32 % 16, index % 32)
            return Float((c * 31 + y * 7 + x * 13) % 256) / 255
        }
        let features = try #require(model.imageFeatures(MLXArray(values, [1, 3, 16, 32])))
        #expect(features.shape == [1, 8, 16])
        let tokens = try EmbeddingGemma2Sequence.tokens(
            [.text([9, 11]), .image(tokens: 8)], configuration: config, beginOfSequence: 2,
            endOfSequence: 1, limit: 64)
        #expect(tokens == [2, 9, 11, 5, 7, 7, 7, 7, 7, 7, 7, 7, 6, 1])
        let output = model.embed(
            inputIds: MLXArray(tokens.map(Int32.init), [1, tokens.count]),
            attentionMask: nil, softTokens: features)
        try MLX.checkedEval(output)
        let expected: [Float] = [
            0.868728220, 0.079042479, -0.071267642, 0.054610837, -0.041429795, -0.136714175,
            -0.154088214, -0.177199394, -0.185128003, -0.179980844, -0.219441086, -0.201574326,
        ]
        #expect(zip(output.asArray(Float.self), expected).allSatisfy { abs($0 - $1) < 0.00001 })
    }

    /// A tiny `gemma4_audio` tower: chunks of 4 over 11 soft tokens, so the attention
    /// window crosses both ends of the sequence.
    private static func tinyAudioConfig(futureContext: Int) -> String {
        #"{"hidden_size":32,"num_hidden_layers":2,"num_attention_heads":4,"rms_norm_eps":1e-06,"output_proj_dims":8,"subsampling_conv_channels":[8,4],"conv_kernel_size":3,"attention_chunk_size":4,"attention_context_left":5,"attention_context_right":\#(futureContext),"attention_logit_cap":30.0,"use_clipped_linears":true}"#
    }

    /// Outputs of the Transformers float32 tower for each `attention_context_right`, with
    /// the deterministic weights and clip bounds that cut about 2% of the activations.
    private static let tinyAudioReference: [Int: [Float]] = [
        0: [
            -0.933767, -0.846326, 0.221115, 0.273132, 0.288333, 0.328264, 0.135564,
            0.087918, -0.696081, -0.802359, 0.182187, 0.181690, 0.169614, 0.206382,
            0.169736, 0.114505, -0.681569, -0.815859, 0.179993, 0.181179, 0.173097,
            0.224357, 0.181873, 0.105458, -0.731858, -0.841028, 0.180635, 0.186791,
            0.174811, 0.224677, 0.194891, 0.134735, -0.299818, -0.706131, 0.084466,
            -0.017602, -0.028063, 0.047058, 0.128417, 0.226803, -0.590354, -0.978634,
            0.209407, 0.148739, 0.113140, 0.153664, 0.172085, 0.149428, -0.460860,
            -1.132344, 0.125645, 0.042860, 0.032337, 0.099797, 0.212641, 0.251729,
            -0.893023, -1.488873, 0.398187, 0.313482, 0.201339, 0.174255, 0.109238,
            0.037691, -0.370202, -1.210834, 0.294910, 0.028567, -0.049218, -0.001000,
            0.096048, 0.158592, -1.278117, -1.653687, 0.555845, 0.481059, 0.357230,
            0.281642, 0.086224, -0.079523, -0.926034, -1.201573, 0.589157, 0.326747,
            0.222999, 0.185851, 0.076138, -0.179193,
        ],
        2: [
            -0.935235, -0.848296, 0.218164, 0.276064, 0.281384, 0.314349, 0.132809,
            0.091375, -0.704404, -0.816061, 0.171046, 0.187457, 0.173699, 0.208407,
            0.173070, 0.114981, -0.671281, -0.839357, 0.174135, 0.182636, 0.174083,
            0.224189, 0.182372, 0.103993, -0.728773, -0.856568, 0.172951, 0.194290,
            0.179700, 0.226396, 0.196834, 0.122967, -0.317566, -0.768354, 0.122657,
            0.004535, -0.011907, 0.060337, 0.137775, 0.187942, -0.594634, -1.002948,
            0.234127, 0.150830, 0.110685, 0.148926, 0.170591, 0.130869, -0.244566,
            -1.113590, 0.154602, -0.079547, -0.105573, -0.033637, 0.136045, 0.300083,
            -0.970510, -1.515017, 0.374081, 0.305015, 0.215908, 0.200127, 0.153027,
            0.102904, -0.031959, -0.935692, 0.252074, -0.321020, -0.422316, -0.368416,
            -0.113295, 0.432087, -1.240948, -1.675985, 0.567147, 0.464822, 0.350919,
            0.277336, 0.080232, -0.097979, -1.008366, -1.170696, 0.641507, 0.325328,
            0.221699, 0.178156, 0.080128, -0.175418,
        ],
    ]

    @Test(arguments: [0, 2])
    func audioTowerMatchesTransformersReference(futureContext: Int) throws {
        let config = try JSONDecoder().decode(
            Gemma4AudioConfiguration.self,
            from: Data(Self.tinyAudioConfig(futureContext: futureContext).utf8))
        let model = Gemma4AudioModel(config: config)
        var weights = try Self.deterministicWeights(for: model)
        // The shared fill would give every clip bound one value and clamp each linear
        // to a constant.
        for key in weights.keys where key.hasSuffix("_min") || key.hasSuffix("_max") {
            let bound: Float = key.contains(".input_") ? 0.9 : 1.25
            weights[key] = MLXArray(key.hasSuffix("_min") ? -bound : bound)
        }
        try model.update(parameters: ModuleParameters.unflattened(weights), verify: .all)
        let mel = MLXArray((0 ..< 41 * 8).map { Float($0 % 13 - 6) * 0.07 }, [1, 41, 8])
        let output = model(mel)
        try MLX.checkedEval(output)
        #expect(output.shape == [1, 11, 8])
        let expected = try #require(Self.tinyAudioReference[futureContext])
        #expect(zip(output.asArray(Float.self), expected).allSatisfy { abs($0 - $1) < 0.00001 })
    }

    /// Writes `processor_config.json` into a new temporary directory.
    private static func processorDirectory(_ json: String) throws -> URL {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        try Data(json.utf8).write(to: directory.appendingPathComponent("processor_config.json"))
        return directory
    }

    /// Reference `Gemma4AudioFeatureExtractor` output for a 4 kHz extractor with 80-sample
    /// frames and 8 mel bins, on two tones and deterministic noise.
    private static let tinyLogMelReference: [Float] = [
        1.31248, 1.95176, 3.13166, 2.48220, 0.72991, 1.48046, 2.72239,
        1.71994, -2.30106, 0.16224, 3.36003, 2.33272, -1.03142, 1.40204,
        2.77141, -0.61510, -2.23433, 0.15352, 3.36799, 2.34194, -1.01509,
        1.39812, 2.78635, -0.61601, -2.18675, 0.17375, 3.36440, 2.33750,
        -1.01892, 1.38831, 2.76945, -0.60327, -2.34667, 0.12412, 3.35855,
        2.33840, -1.00370, 1.39116, 2.75880, -0.62509, -2.26412, 0.08245,
        3.35469, 2.33608, -0.97315, 1.40137, 2.77483, -0.61724, -2.21721,
        0.20032, 3.36333, 2.34128, -1.01380, 1.40629, 2.78734, -0.61443,
        -2.29538, 0.11555, 3.36679, 2.34524, -0.96905, 1.39865, 2.77799,
        -0.62369, -2.33946, 0.17500, 3.36348, 2.33919, -0.95968, 1.37600,
        2.75736, -0.62709, -2.28899, 0.11003, 3.35705, 2.34493, -0.97272,
        1.38441, 2.76407, -0.59484, -2.30095, 0.12808, 3.35628, 2.33730,
        -0.94436, 1.39968, 2.78671, -0.59988,
    ]

    @Test
    func logMelFeaturesMatchTransformersReference() throws {
        let directory = try Self.processorDirectory(
            #"{"feature_extractor":{"feature_size":8,"sampling_rate":4000,"frame_length":80,"hop_length":40,"fft_length":128,"min_frequency":0.0,"max_frequency":2000.0,"mel_floor":0.001,"preemphasis":0.0,"input_scale_factor":1.0,"per_bin_mean":null,"per_bin_stddev":null}}"#
        )
        defer { try? FileManager.default.removeItem(at: directory) }
        let processor = try EmbeddingGemma2AudioProcessor(directory: directory)
        let samples = (0 ..< 450).map { index in
            let time = Double(index) / 4000
            let noise = Double(index * 7919 % 113) / 113 - 0.5
            return Float(
                0.6 * sin(2 * Double.pi * 440 * time) + 0.3 * sin(2 * Double.pi * 1250 * time)
                    + 0.05 * noise)
        }
        let features = processor.features(samples: MLXArray(samples))
        try MLX.checkedEval(features)
        #expect(features.shape == [11, 8])
        #expect(
            zip(features.asArray(Float.self), Self.tinyLogMelReference).allSatisfy {
                abs($0 - $1) < 0.0001
            })
    }

    /// Frame counts of the reference mask, which keeps the first 480,000 samples and
    /// only frames of real samples.
    @Test
    func logMelFramesCoverOnlyRealSamples() throws {
        let directory = try Self.processorDirectory(
            #"{"feature_extractor":{"feature_size":128,"sampling_rate":16000,"frame_length":320,"hop_length":160,"fft_length":512,"min_frequency":0.0,"max_frequency":8000.0,"mel_floor":0.001,"preemphasis":0.0,"input_scale_factor":1.0,"per_bin_mean":null,"per_bin_stddev":null}}"#
        )
        defer { try? FileManager.default.removeItem(at: directory) }
        let processor = try EmbeddingGemma2AudioProcessor(directory: directory)
        for (samples, frames) in [(80, 0), (161, 1), (16_000, 99), (40_000, 249), (656_000, 2_999)]
        {
            let features = processor.features(samples: MLXArray.zeros([samples]))
            #expect(features.shape == [frames, 128])
        }
    }

    /// Expected indices come from the reference `sample_frames`: one frame per second, at
    /// most 32; decoded frames without a frame rate keep every frame before the cap.
    @Test
    func videoSamplingFollowsTheReference() {
        let cases: [(frames: Int, rate: Double?, expected: [Int])] = [
            (300, 30, [0, 30, 60, 90, 120, 150, 180, 210, 240, 270]),
            (
                3000, 25,
                [
                    0, 75, 175, 275, 375, 475, 575, 650, 750, 850, 950, 1050, 1150, 1225, 1325,
                    1425, 1525, 1625, 1725, 1800, 1900, 2000, 2100, 2200, 2300, 2375, 2475, 2575,
                    2675, 2775, 2875, 2975,
                ]
            ),
            (50, 29.97, [0]),
            (241, 23.976, [0, 23, 47, 71, 95, 119, 143, 167, 191, 215]),
            (
                40, nil,
                [
                    0, 1, 2, 3, 5, 6, 7, 8, 10, 11, 12, 13, 15, 16, 17, 18, 20, 21, 22, 23, 25, 26,
                    27, 28, 30, 31, 32, 33, 35, 36, 37, 39,
                ]
            ),
            (5, nil, [0, 1, 2, 3, 4]),
        ]
        for (frames, rate, expected) in cases {
            #expect(
                EmbeddingGemma2VideoProcessor.sampledIndices(
                    frameCount: frames, frameRate: rate, framesPerSecond: 1, maximumFrames: 32)
                    == expected)
        }
    }

    @Test
    func taskPromptsFollowTheModelCard() {
        #expect(
            EmbeddingGemma2Embedding.prompt(text: "hello", title: nil, task: .searchQuery)
                == "task: search result | query: hello")
        #expect(
            EmbeddingGemma2Embedding.prompt(text: "hello", title: nil, task: .document)
                == "title: none | text: hello")
        #expect(
            EmbeddingGemma2Embedding.prompt(
                text: "hello", title: " Quarterly\n Report ", task: .document)
                == "title: Quarterly Report | text: hello")
        #expect(
            EmbeddingGemma2Embedding.prompt(text: "hello", title: nil, task: .clustering)
                == "task: clustering | query: hello")
        #expect(
            EmbeddingGemma2Embedding.prompt(text: "hello", title: nil, task: .classification)
                == "task: classification | query: hello")
        #expect(
            EmbeddingGemma2Embedding.prompt(
                text: "title: Swift | text: hello", title: nil, task: .document)
                == "title: Swift | text: hello")
    }

    /// Drives the real checkpoint end to end — config decode, weight loading, the text
    /// model, the vision and audio towers, and media preparation — against reference
    /// embeddings recorded from the Transformers float32 implementation, the media ones
    /// through Sentence Transformers. Set `EG2_TEST_MODEL_DIR` to a local
    /// `mlx-community/embeddinggemma-2-bf16` snapshot and `EG2_TEST_GOLDEN_DIR` to the
    /// recorded goldens to run.
    @Test(
        .enabled(
            if: ProcessInfo.processInfo.environment["EG2_TEST_MODEL_DIR"] != nil,
            "Requires a local embeddinggemma-2 checkpoint."))
    func downloadedCheckpointMatchesTransformersReference() async throws {
        let modelDirectory = URL(
            filePath: try #require(ProcessInfo.processInfo.environment["EG2_TEST_MODEL_DIR"]))
        let goldenDirectory = URL(
            filePath: ProcessInfo.processInfo.environment["EG2_TEST_GOLDEN_DIR"] ?? "/tmp/eg2ref")
        func file(_ name: String) -> URL { goldenDirectory.appendingPathComponent(name) }
        struct Record: Decodable {
            let image: String?
            let tokens: [Int]
            let embedding: [Float]
        }
        let imageProcessor = try EmbeddingGemma2ImageProcessor(directory: modelDirectory)

        let configData = try Data(contentsOf: modelDirectory.appendingPathComponent("config.json"))
        let config = try JSONDecoder().decode(EmbeddingGemma2Configuration.self, from: configData)
        let model = EmbeddingGemma2(config)
        try await loadWeights(
            modelDirectory: modelDirectory, model: model,
            perLayerQuantization: try JSONDecoder().decode(
                BaseConfiguration.self, from: configData
            ).perLayerQuantization)

        for name in ["golden.json", "vision_golden.json"] {
            let records = try JSONDecoder().decode(
                [Record].self, from: try Data(contentsOf: file(name)))
            for record in records {
                let features = try record.image.map { name in
                    try #require(
                        model.imageFeatures(
                            try imageProcessor.pixels(for: .url(file("\(name).png"))).pixels))
                }
                let vector = model.embed(
                    inputIds: MLXArray(record.tokens.map(Int32.init), [1, record.tokens.count]),
                    attentionMask: nil, softTokens: features)
                try MLX.checkedEval(vector)
                #expect(Self.cosine(vector.asArray(Float.self), record.embedding) > 0.999)
            }
        }

        // Audio, video, and interleaved inputs: the library prepares the media, and the text
        // runs come from the reference tokens, which the rebuilt sequence must equal.
        struct MediaRecord: Decodable {
            let parts: [[String]]
            let tokens: [Int]
            let embedding: [Float]
        }
        let videoProcessor = try EmbeddingGemma2VideoProcessor(directory: modelDirectory)
        let audioProcessor = try EmbeddingGemma2AudioProcessor(directory: modelDirectory)
        let vision = try #require(config.vision)
        let audio = try #require(config.audio)
        let begins: Set = [vision.beginImageTokenID, audio.beginAudioTokenID]
        let records = try JSONDecoder().decode(
            [MediaRecord].self, from: try Data(contentsOf: file("media_golden.json")))
        for record in records {
            var media: [EmbeddingGemma2Sequence.Segment] = []
            var softTokens: [MLXArray] = []
            for part in record.parts where part[0] != "text" {
                let features: MLXArray
                switch part[0] {
                case "image":
                    features = try #require(
                        model.imageFeatures(
                            try imageProcessor.pixels(for: .url(file("\(part[1]).png"))).pixels))
                    media.append(.image(tokens: features.dim(1)))
                case "video":
                    // Movie files sample one frame per second; the decoded clip keeps all.
                    let video: UserInput.Video =
                        part[1] != "clip"
                        ? .url(file(part[1]))
                        : .frames(
                            (0 ..< 5).map {
                                UserInput.VideoFrame(
                                    image: .url(file("frame\($0).png")),
                                    timeStamp: CMTime(value: Int64($0), timescale: 1))
                            })
                    features = try #require(
                        model.imageFeatures(try await videoProcessor.frames(for: video)))
                    media.append(.video(frames: features.dim(0), tokensPerFrame: features.dim(1)))
                default:
                    features = try #require(
                        model.audioFeatures(
                            try await audioProcessor.features(for: .url(file("\(part[1]).wav")))))
                    media.append(.audio(tokens: features.dim(1)))
                }
                softTokens.append(features.reshaped(-1, features.dim(-1)))
            }

            var segments: [EmbeddingGemma2Sequence.Segment] = []
            var index = 1
            while index < record.tokens.count - 1 {
                if begins.contains(record.tokens[index]), !media.isEmpty {
                    let segment = media.removeFirst()
                    segments.append(segment)
                    let length =
                        switch segment {
                        case .video(let frames, let tokens): frames * (tokens + 2)
                        case .image(let tokens), .audio(let tokens): tokens + 2
                        case .text(let text): text.count
                        }
                    index += length
                } else {
                    let end =
                        record.tokens[index ..< record.tokens.count - 1].firstIndex(
                            where: begins.contains) ?? record.tokens.count - 1
                    segments.append(.text(Array(record.tokens[index ..< end])))
                    index = end
                }
            }
            let tokens = try EmbeddingGemma2Sequence.tokens(
                segments, configuration: config, beginOfSequence: 2, endOfSequence: 1,
                limit: EmbeddingGemma2Configuration.contextLength)
            #expect(tokens == record.tokens)

            let vector = model.embed(
                inputIds: MLXArray(tokens.map(Int32.init), [1, tokens.count]), attentionMask: nil,
                softTokens: softTokens.isEmpty ? nil : concatenated(softTokens, axis: 0))
            try MLX.checkedEval(vector)
            #expect(Self.cosine(vector.asArray(Float.self), record.embedding) > 0.999)
        }
    }

    private static func cosine(_ left: [Float], _ right: [Float]) -> Double {
        precondition(left.count == right.count)
        var (dot, leftNorm, rightNorm) = (0.0, 0.0, 0.0)
        for (a, b) in zip(left, right) {
            dot += Double(a) * Double(b)
            leftNorm += Double(a) * Double(a)
            rightNorm += Double(b) * Double(b)
        }
        return dot / (leftNorm.squareRoot() * rightNorm.squareRoot())
    }
}
