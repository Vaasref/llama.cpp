# Development and issue guidelines

This repository is a read-only personal fork of llama.cpp. Pull requests and other code submissions are not accepted. Use the [official llama.cpp repository](https://github.com/ggml-org/llama.cpp) for upstream contribution guidance.

Issues about behavior specific to this fork are welcome. General llama.cpp questions and upstream bugs belong in the official project.

# Reporting issues

Before opening an issue, search this repository for an existing report. Include enough information to reproduce and investigate the problem:

- The exact command and input that trigger the problem
- Expected and actual behavior
- The fork revision and build options
- Relevant logs and error messages
- Operating system, CPU, GPU, driver, and selected backend
- Model and quantization details when they affect the result
- A minimal reproduction or regression test when practical
- Performance or perplexity comparisons for suspected regressions

You are responsible for understanding and checking any code, diagnosis, or test material you provide, regardless of the tools used to produce it.

# Local development and validation

llama.cpp uses the ggml tensor library for model evaluation. If you are unfamiliar with ggml, the [simple](https://github.com/ggml-org/ggml/tree/master/examples/simple), [gpt-2](https://github.com/ggml-org/ggml/tree/master/examples/gpt-2), and [mnist](https://github.com/ggml-org/ggml/tree/master/examples/mnist) examples provide useful introductions.

Keep local changes focused and avoid combining unrelated work. Validate changes in proportion to their scope:

- Run the relevant tests and, when practical, [the full CI suite](ci/README.md).
- Use `llama-perplexity` and `llama-bench` to check accuracy and performance-sensitive changes.
- After modifying `ggml`, run `test-backend-ops` against at least two available backends.
- Add `test-backend-ops` coverage when changing or adding a `ggml` operator.
- Give public API and CLI changes additional compatibility review.
- Start new model or backend work with the smallest testable implementation, then validate each supported backend independently.

New quantization types require additional evidence because they affect storage, accuracy, performance, and every backend:

- Convert a small model to GGUF using the new type.
- Compare perplexity and KL divergence against the native FP16 or BF16 model and similarly sized types.
- Compare performance against similarly sized types with `llama-bench`, including pure CPU results.

# Coding guidelines

- Avoid adding third-party dependencies, extra files, extra headers, etc.
- Always consider cross-compatibility with other operating systems and architectures
- Avoid fancy-looking modern STL constructs, use basic `for` loops, avoid templates, keep it simple
- Vertical alignment makes things more readable and easier to batch edit
- Clean-up any trailing whitespaces, use 4 spaces for indentation, brackets on the same line, `void * ptr`, `int & a`
- Use sized integer types such as `int32_t` in the public API, e.g. `size_t` may also be appropriate for allocation sizes or byte offsets
- Declare structs with `struct foo {}` instead of `typedef struct foo {} foo`
    - In C++ code omit optional `struct` and `enum` keyword whenever they are not necessary
    ```cpp
    // OK
    llama_context * ctx;
    const llama_rope_type rope_type;

    // not OK
    struct llama_context * ctx;
    const enum llama_rope_type rope_type;
    ```

    _(NOTE: this guideline is yet to be applied to the `llama.cpp` codebase. New code should follow this guideline.)_

- Try to follow the existing patterns in the code (indentation, spaces, etc.). In case of doubt use `clang-format` (from clang-tools v15+) to format the added code
- For anything not covered in the current guidelines, refer to the [C++ Core Guidelines](https://isocpp.github.io/CppCoreGuidelines/CppCoreGuidelines)
- Tensors store data in row-major order. We refer to dimension 0 as columns, 1 as rows, 2 as matrices
- Matrix multiplication is unconventional: [`C = ggml_mul_mat(ctx, A, B)`](https://github.com/ggml-org/llama.cpp/blob/880e352277fc017df4d5794f0c21c44e1eae2b84/ggml.h#L1058-L1064) means $C^T = A B^T \Leftrightarrow C = B A^T.$

![matmul](media/matmul.png)

# Naming guidelines

- Use `snake_case` for function, variable and type names
- Naming usually optimizes for longest common prefix (see https://github.com/ggml-org/ggml/pull/302#discussion_r1243240963)

    ```cpp
    // not OK
    int small_number;
    int big_number;

    // OK
    int number_small;
    int number_big;
    ```

- Enum values are always in upper case and prefixed with the enum name

    ```cpp
    enum llama_vocab_type {
        LLAMA_VOCAB_TYPE_NONE = 0,
        LLAMA_VOCAB_TYPE_SPM  = 1,
        LLAMA_VOCAB_TYPE_BPE  = 2,
        LLAMA_VOCAB_TYPE_WPM  = 3,
        LLAMA_VOCAB_TYPE_UGM  = 4,
        LLAMA_VOCAB_TYPE_RWKV = 5,
    };
    ```

- The general naming pattern is `<class>_<method>`, with `<method>` being `<action>_<noun>`

    ```cpp
    llama_model_init();           // class: "llama_model",         method: "init"
    llama_sampler_chain_remove(); // class: "llama_sampler_chain", method: "remove"
    llama_sampler_get_seed();     // class: "llama_sampler",       method: "get_seed"
    llama_set_embeddings();       // class: "llama_context",       method: "set_embeddings"
    llama_n_threads();            // class: "llama_context",       method: "n_threads"
    llama_adapter_lora_free();    // class: "llama_adapter_lora",  method: "free"
    ```

    - The `get` `<action>` can be omitted
    - The `<noun>` can be omitted if not necessary
    - The `_context` suffix of the `<class>` is optional. Use it to disambiguate symbols when needed
    - Use `init`/`free` for constructor/destructor `<action>`

- Use the `_t` suffix when a type is supposed to be opaque to the user - it's not relevant to them if it is a struct or anything else

    ```cpp
    typedef struct llama_context * llama_context_t;

    enum llama_pooling_type llama_pooling_type(const llama_context_t ctx);
    ```

    _(NOTE: this guideline is yet to be applied to the `llama.cpp` codebase. New code should follow this guideline)_

- C/C++ filenames are all lowercase with dashes. Headers use the `.h` extension. Source files use the `.c` or `.cpp` extension
- Python filenames are all lowercase with underscores

- _(TODO: abbreviations usage)_

# Preprocessor directives

- _(TODO: add guidelines with examples and apply them to the codebase)_

    ```cpp
    #ifdef FOO
    #endif // FOO
    ```

# Code maintenance

- Keep large changes focused, understandable, and maintainable.
- Provide the tests, CI coverage, and hardware validation appropriate to the affected backends.
- Record known limitations and keep fork-specific behavior documented.
- New code should follow the guidelines (coding, naming, etc.) outlined in this document. Exceptions are allowed in isolated, backend-specific parts of the code that do not interface directly with the `ggml` interfaces.
  _(NOTE: for legacy reasons, existing code is not required to follow this guideline)_
- For server changes, refer to the [server development documentation](./tools/server/README-dev.md).

# Documentation

- Keep documentation synchronized with fork-specific behavior.
- When an API requires source inspection to understand, document the relevant behavior near its public declaration.
- Report incorrect or outdated documentation as a fork-specific issue.

# Resources

Upstream llama.cpp issues, pull requests, discussions, and project pages remain useful technical references. They are not a submission route for changes made in this fork:

https://github.com/ggml-org/llama.cpp/projects
