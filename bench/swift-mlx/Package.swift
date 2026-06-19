// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "mlx-embed-bench",
    platforms: [.macOS(.v15)],
    dependencies: [
        .package(url: "https://github.com/ml-explore/mlx-swift", from: "0.25.0"),
        .package(url: "https://github.com/ml-explore/mlx-swift-lm", branch: "main"),
        // The #hubDownloader()/#huggingFaceTokenizerLoader() macros expand to code
        // referencing HuggingFace.HubClient and Tokenizers.Tokenizer — the user
        // supplies the HF stack.
        .package(url: "https://github.com/huggingface/swift-huggingface.git", from: "0.9.0"),
        .package(url: "https://github.com/huggingface/swift-transformers", from: "1.3.0"),
    ],
    targets: [
        .executableTarget(
            name: "mlx-embed-bench",
            dependencies: [
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXEmbedders", package: "mlx-swift-lm"),
                .product(name: "MLXHuggingFace", package: "mlx-swift-lm"),
                .product(name: "MLXLMCommon", package: "mlx-swift-lm"),
                .product(name: "HuggingFace", package: "swift-huggingface"),
                .product(name: "Tokenizers", package: "swift-transformers"),
            ],
            path: "Sources"
        )
    ]
)
