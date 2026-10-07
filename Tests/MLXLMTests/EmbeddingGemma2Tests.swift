// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXLMCommon
import MLXNN
import Testing

@testable import MLXVLM

/// Deterministic-checkpoint tests for ``EmbeddingGemma2``: reference vectors from the
/// Transformers float32 implementation, the checkpoint tensor manifest, sequence layout,
/// image resize weights, and task prompts. No weights are downloaded.
struct EmbeddingGemma2Tests {

    /// A tiny text-only config with sliding and global layers of different head sizes.
    private static let tinyConfig =
        #"{"text_config":{"hidden_size":16,"num_hidden_layers":4,"num_attention_heads":2,"num_key_value_heads":2,"head_dim":4,"intermediate_size":32,"hidden_size_per_layer_input":8,"embedding_dim":12,"vocab_size":32,"rms_norm_eps":1e-6,"sliding_window":2,"layer_types":["sliding_attention","full_attention","sliding_attention","full_attention"],"per_layer_config":{"01":{"head_dim":8,"num_key_value_heads":1},"03":{"head_dim":8,"num_key_value_heads":1}},"rope_parameters":{"full_attention":{"rope_theta":1000000.0,"rope_type":"default"},"sliding_attention":{"rope_theta":10000.0,"rope_type":"default"}}}}"#

    /// The tiny config plus a small vision encoder and image tokens.
    private static let tinyVisionConfig =
        #"{"text_config":{"hidden_size":16,"num_hidden_layers":2,"num_attention_heads":2,"num_key_value_heads":1,"head_dim":8,"intermediate_size":32,"hidden_size_per_layer_input":8,"embedding_dim":12,"vocab_size":32,"rms_norm_eps":1e-06,"sliding_window":4,"layer_types":["sliding_attention","full_attention"],"per_layer_config":{"01":{"head_dim":8,"num_key_value_heads":1}},"rope_parameters":{"full_attention":{"rope_theta":1000000.0,"rope_type":"default"},"sliding_attention":{"rope_theta":10000.0,"rope_type":"default"}}},"vision_config":{"model_type":"gemma4_vision","hidden_size":32,"intermediate_size":64,"num_hidden_layers":2,"num_attention_heads":2,"num_key_value_heads":2,"head_dim":16,"patch_size":4,"pooling_kernel_size":2,"position_embedding_size":64,"default_output_length":9,"rms_norm_eps":1e-06,"use_clipped_linears":false,"standardize":false,"rope_parameters":{"rope_theta":100.0,"rope_type":"axial"}},"image_token_id":7,"boi_token_id":5,"eoi_token_id":6}"#

    /// `text_config`, `vision_config`, and image tokens of `google/embeddinggemma-2`.
    private static let checkpointConfig =
        #"{"text_config":{"attention_bias":false,"attention_dropout":0.0,"bos_token_id":2,"dtype":"bfloat16","embedding_dim":768,"eos_token_id":1,"head_dim":256,"hidden_activation":"gelu_pytorch_tanh","hidden_size":512,"hidden_size_per_layer_input":512,"initializer_range":0.02,"intermediate_size":2048,"layer_types":["sliding_attention","sliding_attention","sliding_attention","sliding_attention","sliding_attention","full_attention","sliding_attention","sliding_attention","sliding_attention","sliding_attention","sliding_attention","full_attention","sliding_attention","sliding_attention","sliding_attention","sliding_attention","sliding_attention","full_attention","sliding_attention","sliding_attention","sliding_attention","sliding_attention","sliding_attention","full_attention"],"max_position_embeddings":262144,"model_type":"embedding_gemma2_text","num_attention_heads":4,"num_hidden_layers":24,"num_key_value_heads":2,"pad_token_id":0,"per_layer_config":{"05":{"head_dim":512,"num_key_value_heads":1},"11":{"head_dim":512,"num_key_value_heads":1},"17":{"head_dim":512,"num_key_value_heads":1},"23":{"head_dim":512,"num_key_value_heads":1}},"rms_norm_eps":1e-06,"rope_parameters":{"full_attention":{"rope_theta":1000000.0,"rope_type":"default"},"sliding_attention":{"rope_theta":10000.0,"rope_type":"default"}},"sliding_window":512,"vocab_size":262144},"vision_config":{"model_type":"gemma4_vision","hidden_size":768,"intermediate_size":3072,"num_hidden_layers":16,"num_attention_heads":12,"num_key_value_heads":12,"head_dim":64,"patch_size":16,"pooling_kernel_size":3,"position_embedding_size":10240,"default_output_length":280,"rms_norm_eps":1e-06,"use_clipped_linears":false,"standardize":false,"rope_parameters":{"rope_theta":100.0,"rope_type":"axial"}},"image_token_id":258880,"boi_token_id":255999,"eoi_token_id":258882}"#

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
    func checkpointConfigurationResolvesPerLayerAttentionAndVision() throws {
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
        #expect(vision.encoder.hiddenSize == 768)
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

    /// Names and shapes of every text and vision tensor in the `google/embeddinggemma-2`
    /// safetensors header, through the module tree the loader updates.
    @Test
    func modulesMatchTheCheckpointTextAndVisionTensors() throws {
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
        let actual = Dictionary(
            uniqueKeysWithValues: model.parameters().flattened().map { (name, value) in
                (name, value.shape)
            })
        #expect(actual.count == 413 + 211)
        #expect(actual == expected)
    }

    @Test(arguments: [true, false])
    func sanitizeKeepsTheLoadedTowersAndDropsAudio(_ hasVision: Bool) throws {
        let tensor = MLXArray.zeros([1])
        let model = EmbeddingGemma2(
            try Self.decode(hasVision ? Self.tinyVisionConfig : Self.tinyConfig))
        let clean = try model.sanitize(
            weights: [
                "language_model.norm.weight": tensor,
                "vision_tower.patch_embedder.input_proj.weight": tensor,
                "embed_vision.embedding_projection.weight": tensor,
                "audio_tower.output_proj.weight": tensor,
                "embed_audio.embedding_projection.weight": tensor,
            ])
        let vision: Set<String> = [
            "vision_tower.patch_embedder.input_proj.weight",
            "embed_vision.embedding_projection.weight",
        ]
        #expect(
            Set(clean.keys)
                == (hasVision
                    ? vision.union(["language_model.norm.weight"]) : ["language_model.norm.weight"])
        )
    }

    @Test
    func imageSequenceKeepsImagesAndTruncatesText() throws {
        let vision = try #require(try Self.decode(Self.tinyVisionConfig).vision)
        #expect(
            try EmbeddingGemma2Sequence.tokens(
                text: [2], imageTokenCounts: [2, 1], vision: vision, endOfSequence: 1, limit: 16)
                == [2, 5, 7, 7, 6, 5, 7, 6, 1])
        #expect(
            try EmbeddingGemma2Sequence.tokens(
                text: [2, 9, 10, 11], imageTokenCounts: [2], vision: vision, endOfSequence: 1,
                limit: 7)
                == [2, 9, 5, 7, 7, 6, 1])
        #expect(throws: EmbeddingGemma2Embedding.Error.contextExceeded) {
            try EmbeddingGemma2Sequence.tokens(
                text: [2], imageTokenCounts: [5], vision: vision, endOfSequence: 1, limit: 8)
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
            text: [2, 9, 11], imageTokenCounts: [8], vision: try #require(config.vision),
            endOfSequence: 1, limit: 64)
        #expect(tokens == [2, 9, 11, 5, 7, 7, 7, 7, 7, 7, 7, 7, 6, 1])
        let output = model.embed(
            inputIds: MLXArray(tokens.map(Int32.init), [1, tokens.count]),
            attentionMask: nil, imageFeatures: features)
        try MLX.checkedEval(output)
        let expected: [Float] = [
            0.868728220, 0.079042479, -0.071267642, 0.054610837, -0.041429795, -0.136714175,
            -0.154088214, -0.177199394, -0.185128003, -0.179980844, -0.219441086, -0.201574326,
        ]
        #expect(zip(output.asArray(Float.self), expected).allSatisfy { abs($0 - $1) < 0.00001 })
    }

    /// The vision tower batches only equal sizes, so each image is encoded alone and the
    /// soft tokens are joined in input order.
    @Test
    func imageFeaturesJoinDifferentSizesInOrder() throws {
        let model = EmbeddingGemma2(try Self.decode(Self.tinyVisionConfig))
        try model.update(
            parameters: ModuleParameters.unflattened(try Self.deterministicWeights(for: model)),
            verify: .all)
        let wide = Self.pixels(height: 16, width: 32)
        let square = Self.pixels(height: 16, width: 16)
        let wideFeatures = try #require(model.imageFeatures(wide))
        let squareFeatures = try #require(model.imageFeatures(square))
        let joined = try #require(model.imageFeatures([wide, square]))
        try MLX.checkedEval(wideFeatures, squareFeatures, joined)
        #expect(wideFeatures.shape == [1, 8, 16])
        #expect(squareFeatures.shape == [1, 4, 16])
        #expect(joined.shape == [1, 12, 16])
        let expected = wideFeatures.asArray(Float.self) + squareFeatures.asArray(Float.self)
        #expect(zip(joined.asArray(Float.self), expected).allSatisfy { abs($0 - $1) < 0.00001 })
    }

    private static func pixels(height: Int, width: Int) -> MLXArray {
        let plane = height * width
        let values: [Float] = (0 ..< (3 * plane)).map { index in
            let (c, y, x) = (index / plane, index / width % height, index % width)
            return Float((c * 31 + y * 7 + x * 13) % 256) / 255
        }
        return MLXArray(values, [1, 3, height, width])
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

    /// Drives the real checkpoint end to end — config decode, weight loading, text model,
    /// and vision tower — against reference embeddings recorded from the Transformers
    /// float32 implementation. Set `EG2_TEST_MODEL_DIR` to a local
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
        struct Record: Decodable {
            let image: String?
            let tokens: [Int]
            let embedding: [Float]
        }
        let imageProcessor = try EmbeddingGemma2ImageProcessor(directory: modelDirectory)

        let config = try JSONDecoder().decode(
            EmbeddingGemma2Configuration.self,
            from: try Data(contentsOf: modelDirectory.appendingPathComponent("config.json")))
        let model = EmbeddingGemma2(config)
        try await loadWeights(modelDirectory: modelDirectory, model: model)

        for name in ["golden.json", "vision_golden.json"] {
            let records = try JSONDecoder().decode(
                [Record].self,
                from: try Data(contentsOf: goldenDirectory.appendingPathComponent(name)))
            for record in records {
                let features = try record.image.map { name in
                    try #require(
                        model.imageFeatures(
                            try imageProcessor.pixels(
                                for: .url(goldenDirectory.appendingPathComponent("\(name).png"))
                            ).pixels))
                }
                let vector = model.embed(
                    inputIds: MLXArray(record.tokens.map(Int32.init), [1, record.tokens.count]),
                    attentionMask: nil, imageFeatures: features)
                try MLX.checkedEval(vector)
                let actual = vector.asArray(Float.self)
                #expect(actual.count == record.embedding.count)
                var dot = 0.0
                var left = 0.0
                var right = 0.0
                for (a, b) in zip(actual, record.embedding) {
                    dot += Double(a) * Double(b)
                    left += Double(a) * Double(a)
                    right += Double(b) * Double(b)
                }
                #expect(dot / (left.squareRoot() * right.squareRoot()) > 0.999)
            }
        }
    }
}
